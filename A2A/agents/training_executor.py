import asyncio
import os
import json
import torch
from typing import Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from datasets import Dataset

from python_a2a.models.message import Message, MessageRole
from python_a2a.utils.conversion import create_text_message
from python_a2a.server.base import BaseA2AServer
from utils.training_utils import load_data, preprocess_for_causal_lm

# --- Configuration Models ---

class StrategyType(str, Enum):
    FULL_TRAINING = "full_training"
    FINE_TUNING = "fine_tuning"
    TRANSFER = "transfer"
    LORA = "lora"
    ADAPTER = "adapter"
    FREEZING = "layer_freezing"
    CONTINUAL = "continual"
    CURRICULUM = "curriculum"

class TrainingConfig(BaseModel):
    model_name: str
    dataset_path: str
    output_dir: str = "./output"
    strategy: StrategyType
    epochs: float = 1.0
    batch_size: int = 4
    learning_rate: float = 2e-5
    # Optional params
    lora_r: int = 8
    lora_alpha: int = 32
    target_modules: Optional[list] = None # e.g. ["q_proj", "v_proj"]
    freeze_layers: int = 0
    continual_tasks: Optional[list] = None # List of dataset paths for sequential training

# --- Strategy Implementations ---

class BaseStrategy:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        return model

    def prepare_dataset(self, path: str):
        ds = load_data(path)
        tokenized_ds = ds.map(
            lambda x: preprocess_for_causal_lm(x, self.tokenizer),
            batched=True,
            remove_columns=ds.column_names
        )
        return tokenized_ds

    def train(self, model, dataset):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=50,
            evaluation_strategy="no",
            use_cpu=not torch.cuda.is_available()
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        trainer.train()
        trainer.save_model(self.config.output_dir)
        return f"Training complete. Model saved to {self.config.output_dir}"

class StandardStrategy(BaseStrategy):
    """Full Training and Fine-Tuning"""
    pass

class PeftStrategy(BaseStrategy):
    """LoRA and Adapters"""
    def prepare_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        if self.config.strategy == StrategyType.LORA:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=self.config.lora_r, 
                lora_alpha=self.config.lora_alpha, 
                lora_dropout=0.1,
                target_modules=self.config.target_modules
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # Add Adapter logic here if needed separate from LoRA, but LoRA is most common
        
        return model

class LayerFreezingStrategy(BaseStrategy):
    def prepare_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # Simple freezing of first N layers
        # Transformer models usually have 'transformer' or 'model' attribute with layers
        # This is model-specific logic, implementing a generic heuristic
        
        layers_to_freeze = self.config.freeze_layers
        if layers_to_freeze > 0:
            # Attempt to find the layer list. GPT-2 -> h, BERT -> layer, etc.
            # Generic approach: freeze parameters that contain 'layer.0' through 'layer.N'
            # Or just freeze embeddings and first N blocks
            
            count = 0
            for name, param in model.named_parameters():
                # naive heuristic check
                if "embed" in name:
                    param.requires_grad = False
                
                # Check for layer index in name
                # e.g. transformer.h.0.attn...
                parts = name.split('.')
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        if layer_idx < layers_to_freeze:
                            param.requires_grad = False
                        break
                        
        return model

class ContinualStrategy(BaseStrategy):
    """Sequential training on multiple tasks"""
    
    def train(self, model, dataset):
        # In continual strategy, 'dataset' arg might be just the first one, 
        # but we rely on config.continual_tasks list
        
        tasks = self.config.continual_tasks or [self.config.dataset_path]
        
        output_dir_base = self.config.output_dir
        
        report = []
        
        for i, task_path in enumerate(tasks):
            print(f"--- Continual Learning: Task {i+1}/{len(tasks)}: {task_path} ---")
            
            # Update output dir for this task
            self.config.output_dir = f"{output_dir_base}/task_{i+1}"
            
            # Prepare dataset for this task
            task_ds = self.prepare_dataset(task_path)
            
            # EWC logic could be added here (calculating Fisher info from previous task)
            # For this MVP, we implement Naive Rehearsal / Finetuning sequence
            
            msg = super().train(model, task_ds)
            report.append(f"Task {i+1}: {msg}")
            
        return "\n".join(report)

class CurriculumStrategy(BaseStrategy):
    def prepare_dataset(self, path: str):
        ds = load_data(path)
        
        # Simple Curriculum: Sort by length (shorter first = easier?)
        # Or if there is a 'difficulty' column
        
        if "text" in ds.column_names:
            # Sort by text length
            # Note: Dataset object sort in recent HF datasets
            ds = ds.map(lambda x: {"length": len(x["text"])})
            ds = ds.sort("length")
            
        tokenized_ds = ds.map(
            lambda x: preprocess_for_causal_lm(x, self.tokenizer),
            batched=True,
            remove_columns=ds.column_names
        )
        return tokenized_ds

# --- Executor ---

class TrainingExecutor(BaseA2AServer):
    def __init__(self, mcp_server_path: str = ""):
        # MCP not strictly used here yet, but keeping signature consistent
        pass

    async def execute_task(self, message: Message) -> Message:
        try:
            content_obj = message.content
            args = {}
            if hasattr(content_obj, 'text'):
                try:
                    args = json.loads(content_obj.text)
                except: pass
            
            config = TrainingConfig(**args)
            
            strategy_map = {
                StrategyType.FULL_TRAINING: StandardStrategy,
                StrategyType.FINE_TUNING: StandardStrategy,
                StrategyType.TRANSFER: StandardStrategy, # Simplified to standard for MVP
                StrategyType.LORA: PeftStrategy,
                StrategyType.ADAPTER: PeftStrategy,
                StrategyType.FREEZING: LayerFreezingStrategy,
                StrategyType.CONTINUAL: ContinualStrategy,
                StrategyType.CURRICULUM: CurriculumStrategy
            }
            
            strategy_cls = strategy_map.get(config.strategy)
            if not strategy_cls:
                return create_text_message(f"Unknown strategy: {config.strategy}", role=MessageRole.SYSTEM)
                
            runner = strategy_cls(config)
            print(f"[TrainingExecutor] Preparing model for {config.strategy}...")
            model = runner.prepare_model()
            
            # For continual, dataset preparation happens in loop, but need initial one for API consistency
            dataset = runner.prepare_dataset(config.dataset_path) if config.strategy != StrategyType.CONTINUAL else None
            
            print(f"[TrainingExecutor] Starting training...")
            result = runner.train(model, dataset)
            
            return create_text_message(result, role=MessageRole.AGENT)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return create_text_message(f"Training failed: {str(e)}", role=MessageRole.SYSTEM)

    def handle_message(self, message: Message) -> Message:
        return asyncio.run(self.execute_task(message))
