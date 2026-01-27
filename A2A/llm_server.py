import os
import uvicorn
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
        
        # Check if local path exists, if not let transformers download it (or fail if no internet/auth)
        if os.path.exists(model_path):
            print(f"[LLM Service] Loading from local path: {model_path}")
        else:
             print(f"[LLM Service] Local path {model_path} not found. Attempting to download/load from Hub...")
             # fallback to a default if env var points to nothing valid? 
             # For now assume the user handles model placement or valid hub ID
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("[LLM Service] Model loaded successfully.")
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
        
        # Explicitly set max_new_tokens to avoid conflict with max_length if using pipeline defaults
        # transformers pipeline may use max_length as total length (prompt + new), 
        # while user might expect it to vary. 
        # For simplicity with pipeline, we just pass max_new_tokens if that's what we want, 
        # or rely on max_length. 
        # The warning said max_new_tokens=256 was set by default ? 
        # Let's use max_new_tokens explicitly.
        
        results = generator(
            request.prompt, 
            max_new_tokens=request.max_length, # Map user's max_length to max_new_tokens for clarity
            do_sample=True, 
            temperature=request.temperature
        )
        
        elapsed = time.time() - start_time
        print(f"[LLM Service] Generation took {elapsed:.2f}s")
        
        summary = results[0]['generated_text']
        
        # Strip prompt if included (transformers usually includes it)
        if summary.startswith(request.prompt):
            summary = summary[len(request.prompt):].strip()
            
        return {"text": summary}
    except Exception as e:
        print(f"[LLM Service] Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": generator is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
