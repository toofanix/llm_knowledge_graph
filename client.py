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

        with requests.post(url, json=payload, stream=False) as response:
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

# Create a model from a Modelfile.
def create(model_name, model_path, callback=None):
    try:
        url = f"{BASE_URL}/api/create"
        payload = {"name": model_name, "path": model_path}

        # Make a POST request with the stream=True 
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # Iterating over the response line and line
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                if callback:
                    callback(chunk)
                else:
                    print(f"Status: {chunk.get('status')}")
    except requests.exceptions.RequestException as e:
        print(f"An error occured: {e}")


# Pull a model from model reistry. Cancelled pulls a resumed 
# and multiple call will share the same download progress. 
def pull(model_name, insecure=False, callback=None):
    try:
        url = f"{BASE_URL}/api/pull"
        payload = {"name": model_name, "insecure":insecure}

        # POST request with stream=True
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # Iterate over response line by line
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)

                if callback:
                    callback(line)
                else:
                    print(chunk.get('status', ''), end='', flush=True)

                # If there is layer data, print that
                if 'digest' in chunk:
                    print(f" - Digest: {chunk['digest']}", end='', flush=True)
                    print(f" - Total: {chunk['total']}", end='', flush=True)
                    print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                else:
                    print()
    except requests.exceptions.RequestException as e:
        print(f"An error occured: {e}")

# Push a model to model registry.
def push(model_name, insecure=False, callback=None):
    try:
        url = f"{BASE_URL}/api/push"
        payload = {"name": model_name, "insecure": insecure}

        # Make a POST request with stream=True
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            #Iterate over the response line by line
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                if callback:
                    callback(line)
                else:
                    print(chunk.get('status', ''), end='', flush=True)
                
                # If there is layer data, print it
                if 'digest' in chunk:
                    print(f" - Digest: {chunk['digest']}", end='', flush=True)
                    print(f" - Total: {chunk['total']}", end='', flush=True)
                    print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                else:
                    print()
    except requests.exceptions.RequestException as e:
        print(f"An error occured: {e}")

# List models that are available locally
def list():
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get('model', [])
        return models

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
# Copy a model. Creates a model with another name from an existing model.
def copy(source, destination):
    try:
        # Create a JSON payload
        payload = {"source": source, "destination": destination}

        response = requests.post(f"{BASE_URL}/api/copy", json=payload)
        response.raise_for_status()
        return "Copy successful"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    

# delete a model and its data    
def delete(model_name):
    try:
        payload = {"name": model_name}
        response = requests.post(f"{BASE_URL}/api/delete", json=payload)
        response.raise_for_status()
        return "Delete successful"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
# Show info for a model
def show(model_name):
    try:
        payload = {"name": model_name}
        response = requests.post(f"{BASE_URL}/api/show", json=payload)
        response.raise_for_status()

        # Parse the response in to json
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def check_status():
    try:
        response = requests.head(f"{BASE_URL}/")
        response.raise_for_status()
        return "OLLAMA is running"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return "Ollama is not running"
    
    