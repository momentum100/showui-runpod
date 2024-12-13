import runpod
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor

load_dotenv()

# --- Model Loading (Optimized) ---
QWENVL_MODEL_ID = "Qwen/Qwen-VL-Chat"
SHOWUI_MODEL_ID = "showlab/ShowUI-2B"

try:
    # Load Qwen-VL-Chat first
    qwen_processor = AutoProcessor.from_pretrained(QWENVL_MODEL_ID, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWENVL_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    # Load ShowUI on top of Qwen
    showui_processor = qwen_processor # The same processor can be used
    showui_model = AutoModelForCausalLM.from_pretrained(
        SHOWUI_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# --- Image Preprocessing ---
def preprocess_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

# --- Inference Function ---
def inference(image):
    try:
        # Use the Qwen-VL-Chat model for initial processing
        query = "What is this image about?" # Customize prompt as needed
        inputs = qwen_processor(text=query, images=[image], return_tensors='pt').to("cuda", torch.bfloat16) # Qwen expects a list of images
        
        qwen_outputs = qwen_model.generate(**inputs, max_new_tokens=512)
        qwen_response = qwen_processor.decode(qwen_outputs[0], skip_special_tokens=False)

        # Extract text and image from Qwen response for ShowUI input
        extracted_text = qwen_response.split("<|im_end|>")[0].split("Assistant:")[1].strip()  # Adjust based on actual Qwen output
        # The image remains the same, since it was already embedded by Qwen

        # Feed the output into the ShowUI model
        showui_inputs = showui_processor(
            text=extracted_text + "[<IMG>]{{}}[/<IMG>]",  # Construct prompt for ShowUI
            images=image,  # Use the original preprocessed image
            return_tensors="pt",
        ).to("cuda", torch.bfloat16)

        showui_outputs = showui_model.generate(**showui_inputs, max_new_tokens=512)
        showui_response = showui_processor.batch_decode(showui_outputs, skip_special_tokens=True)[0].strip()

        return {
            "qwen_response": qwen_response,  # Optionally, return the Qwen output
            "showui_response": showui_response
        }
    except Exception as e:
        return {"error": str(e)}

# --- RunPod Handler ---
def handler(event):
    try:
        input_data = event['input']
        image_base64 = input_data.get('image')

        if not image_base64:
            return {"error": "No image provided"}

        image = preprocess_image(image_base64)
        result = inference(image)
        return result

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
