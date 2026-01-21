import unittest
import ollama
import time
import sys

class TestLLMServing(unittest.TestCase):
    def setUp(self):
        """
        Check if Ollama is reachable.
        """
        try:
            # Simple check to see if we can list models
            ollama.list()
        except Exception as e:
            self.skipTest(f"Ollama is not reachable: {e}. Make sure 'ollama serve' is running.")

    def test_01_fetch_model(self):
        """
        Test fetching (pulling) the Gemma model.
        This might take time if the model is not cached.
        """
        model_name = "gemma:2b"
        print(f"\n>>> [Test] Pulling model {model_name}... (This may take time)")
        
        # Pull the model. verify success by checking if it exists in list after pull
        # ollama.pull streams progress, ensuring it completes without error is the goal
        try:
            progress_generator = ollama.pull(model_name, stream=True)
            for progress in progress_generator:
                status = progress.get('status', '')
                # Print status sparingly to avoid cluttering logs, or just last status
                if 'completed' in status or 'downloading' in status:
                    sys.stdout.write(f"\r{status}")
                    sys.stdout.flush()
            print("\n>>> [Test] Model pull complete.")
        except Exception as e:
            self.fail(f"Failed to pull model {model_name}: {e}")

        # Verify it's in the list
        models = ollama.list()
        # models['models'] is a list of objects usually.
        # Check for 'model' or 'name' field. 
        model_names = []
        for m in models.get('models', []):
             # Handle dict or object
             if isinstance(m, dict):
                 model_names.append(m.get('model') or m.get('name'))
             else:
                 # Assume object with attributes, access as dict might fail if not implemented fully? 
                 # But traceback showed __getitem__, so it tries.
                 # Updated check:
                 model_names.append(getattr(m, 'model', getattr(m, 'name', str(m))))
        
        # Allow exact match or with tag
        found = any(model_name in name for name in model_names)
        self.assertTrue(found, f"Model {model_name} not found in ollama list after pull.")

    def test_02_serve_model(self):
        """
        Test serving (chatting) with the Gemma model.
        """
        model_name = "gemma:2b"
        prompt = "Hello, are you working correctly? Reply with 'Yes, I am operational.'"
        
        print(f"\n>>> [Test] Sending prompt to {model_name}...")
        
        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            content = response['message']['content']
            print(f">>> [Response] {content}")
            
            self.assertTrue(len(content) > 0, "Received empty response from model.")
            # We don't strictly check for the exact string as LLMs vary, but checking for non-empty is good.
            
        except Exception as e:
            self.fail(f"Failed to chat with model {model_name}: {e}")

if __name__ == '__main__':
    unittest.main()
