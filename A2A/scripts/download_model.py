import os
import argparse
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

def download_model(model_id, save_dir, endpoint=None, token=None):
    """
    Download a model snapshot from Hugging Face or a mirror.
    """
    print(f"Downloading model {model_id} to {save_dir}...")
    if endpoint:
        print(f"Using custom endpoint: {endpoint}")
        os.environ["HF_ENDPOINT"] = endpoint
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=save_dir,
            local_dir_use_symlinks=False,  # Important for portability/offline use
            token=token
        )
        print(f"Successfully downloaded {model_id} to {save_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face model for local usage.")
    parser.add_argument("--model_id", type=str, default="google/gemma-2b-it", help="The model ID on Hugging Face.")
    parser.add_argument("--save_dir", type=str, default="./models/gemma-2b-it", help="Local directory to save the model.")
    parser.add_argument("--endpoint", type=str, help="Optional custom HF endpoint (e.g., internal mirror).")
    parser.add_argument("--token", type=str, help="Hugging Face token if required (or set HF_TOKEN env var).")

    args = parser.parse_args()
    
    load_dotenv()
    download_model(args.model_id, args.save_dir, args.endpoint, args.token)
