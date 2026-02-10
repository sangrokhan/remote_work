import os
import uvicorn
import uuid
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 500
    temperature: float = 0.7

# Global model and tokenizer
model = None
tokenizer = None
generator = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer, generator
    print("[LLM Service] Loading model...")
    try:
        model_path = os.environ.get("MODEL_PATH", "./models/gemma-2-2b-it")
        
        # Check device availability
        use_cuda = torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available()
        
        print(f"[LLM Service] Device Check - CUDA: {use_cuda}, MPS: {use_mps}")
        
        device = "cpu"
        if use_cuda:
            device = "cuda"
        elif use_mps:
            device = "mps"
            
        print(f"[LLM Service] Selected device: {device}")
        
        # Check if local path exists
        if os.path.exists(model_path):
            print(f"[LLM Service] Loading from local path: {model_path}")
        else:
             print(f"[LLM Service] Local path {model_path} not found. Attempting to download/load from Hub...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        
        if device != "cpu":
            print(f"[LLM Service] Moving model to {device}...")
            model.to(device)
            
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print(f"[LLM Service] Model loaded successfully on {model.device}.")
    except Exception as e:
        print(f"[LLM Service] Error loading model: {e}")
        # We might not want to crash immediately to allow debugging, but the service is useless without model
        raise e

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"[LLM Service] Generating for prompt: {request.prompt[:50]}...")
        import time
        start_time = time.time()
        
        # Avoid passing individual params with generation_config
        results = generator(
            request.prompt, 
            max_new_tokens=request.max_length,
            do_sample=True, 
            temperature=request.temperature,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=None
        )
        
        elapsed = time.time() - start_time
        print(f"[LLM Service] Generation took {elapsed:.2f}s")
        
        summary = results[0]['generated_text']
        if summary.startswith(request.prompt):
            summary = summary[len(request.prompt):].strip()
            
        return {"text": summary}
    except Exception as e:
        print(f"[LLM Service] Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenAI Compatible API ---

@app.get("/v1/models")
async def list_models():
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    return {
        "object": "list",
        "data": [
            {
                "id": model_path,
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner"
            }
        ]
    }

@app.get("/v1/model")
async def get_model_legacy():
    return await list_models()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 500
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Simple chat template fallback
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "

    try:
        import time
        results = generator(
            prompt,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=None
        )
        
        text = results[0]['generated_text']
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
            
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
    except Exception as e:
        print(f"[LLM Service] Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    device_info = "model not loaded"
    if generator is not None:
        device_info = str(model.device)
    return {
        "status": "ok", 
        "model_loaded": generator is not None,
        "device": device_info
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
