import os
import json
import requests

BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Generate a response for a given prompt with a provided model.
def generate(model_name, prompt, system=None, template=None, context=None, options=None, callback=None):
    try:
        url = f"{BASE_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system,
            "template": template,
            "context": context,
            "options": options
        }

        # Remove keys with None values
        payload = {k: v for k, v in payload.items() if v is None}

        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # variable to hold the context history of final chunk
            final_context = None

            # Variable to hold concatenated response strings if no callback
            full_response = ""

            # Iterating over the response line by line 
            for line in response.iter_lines():
                if line:
                    # parse each line (json chunk) and extract the details
                    chunk = json.loads(line)

                    # If a callback function is provided, call it with the chunk
                    if callback:
                        callback(chunk)
                    else:
                        # If this is not last chunk, add the "response" field value to full_response and print it
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                            print(response_piece, end="", flush=True)
                    
                    if chunk.get("done"):
                        final_context = chunk.get("context")
            # Return the full response nd the final context
            return full_response, final_context
        
    except requests.exceptions.RequestException as e:
        print(f"An error occured: {e}")
        return None, None

