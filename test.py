import sys
import requests
import base64
import json
from pathlib import Path
import os
import time
from dotenv import load_dotenv

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_request_to_runpod(endpoint_id, prompt, image_path):
    # Validate inputs
    if not all([endpoint_id, prompt, image_path]):
        print("Usage: python test.py <endpoint_id> <prompt> <image_path>")
        sys.exit(1)

    # Validate image path
    if not Path(image_path).is_file():
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    # RunPod API endpoint
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    # Convert image to base64
    try:
        image_base64 = encode_image_to_base64(image_path)
    except Exception as e:
        print(f"Error encoding image: {e}")
        sys.exit(1)

    # Prepare payload
    payload = {
        "input": {
            "image": image_base64,
            "query": prompt
        }
    }

    # Headers
    headers = {
        'Authorization': f'Bearer {RUNPOD_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        # Send POST request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Get the task ID from the response
        task_id = response.json().get('id')
        print(f"Request submitted successfully. Task ID: {task_id}")

        # Poll for results
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{task_id}"
        while True:
            status_response = requests.get(status_url, headers=headers)
            status_data = status_response.json()
            
            if status_data.get('status') == 'COMPLETED':
                print("\nResults:")
                print(json.dumps(status_data.get('output'), indent=2))
                break
            elif status_data.get('status') == 'FAILED':
                print(f"\nTask failed: {status_data.get('error')}")
                break
            
            print(".", end="", flush=True)
            time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test.py <endpoint_id> <prompt> <image_path>")
        sys.exit(1)

    # Load environment variables from .env file
    load_dotenv()
    
    # Get RunPod API key from .env
    RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY not found in .env file")
        sys.exit(1)

    endpoint_id = sys.argv[1]
    prompt = sys.argv[2]
    image_path = sys.argv[3]

    send_request_to_runpod(endpoint_id, prompt, image_path) 