import runpod
import torch
from PIL import Image
import io
import base64
import os
import ast
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
    showui_processor = AutoProcessor.from_pretrained(SHOWUI_MODEL_ID, trust_remote_code=True)
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
def inference(image, query):
    try:
        # --- UI Grounding Prompt ---
        _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM},
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},  # User's query about location
                ],
            }
        ]

        text = showui_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = showui_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to("cuda", torch.bfloat16)

        generated_ids = showui_model.generate(**inputs, max_new_tokens=128)

        output_text = showui_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract coordinates using ast.literal_eval
        try:
            click_xy = ast.literal_eval(output_text)  # Parse the output string as a Python literal
        except (ValueError, SyntaxError) as e:
            return {"error": f"Could not extract coordinates from output: {output_text}, Error: {e}"}

        return {
            "coordinates": click_xy,
            "debug_output": output_text,  # For debugging purposes
        }

    except Exception as e:
        return {"error": str(e)}

# --- RunPod Handler ---
def handler(event):
    try:
        input_data = event['input']
        image_base64 = input_data.get('image')
        query = input_data.get('query')

        if not image_base64:
            return {"error": "No image provided"}
        if not query:
            return {"error": "No query provided"}

        image = preprocess_image(image_base64)
        result = inference(image, query)  # Pass the query to the inference function
        return result

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
