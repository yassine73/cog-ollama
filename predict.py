from cog import BasePredictor, Input, ConcatenateIterator
import json
import time
import requests
import subprocess

MODEL_NAME = "salmatrafi/acegpt:13b"
OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_GENERATE = OLLAMA_API + "/api/generate"

class Predictor(BasePredictor):
    def setup(self):
        """Setup necessary resources for predictions"""
        # Start server
        print("Starting ollama server")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(2)
        print("Downloading model")
        subprocess.check_call(["ollama", "pull", MODEL_NAME], close_fds=False)
        # Load model
        print("Running model")
        subprocess.check_call(["ollama", "run", MODEL_NAME], close_fds=False)

    def predict(
            self, prompt: str = Input(description="Input text for the model"),
            temperature: float = Input(description="Temperature", default=0.7),
            max_tokens: int = Input(description="Temperature", default=30)
        ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model and stream the output"""
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "options" : {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        with requests.post(
            OLLAMA_GENERATE,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
            timeout=60
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            yield chunk['response']
                    except json.JSONDecodeError:
                        print("Failed to parse response chunk as JSON")
        
        end_time = time.time()
        total_time = end_time - start_time
        print("Total runtime:", total_time)