import runpod
import torch
from PIL import Image
import io
import base64
import os
import ast
import logging
from dotenv import load_dotenv
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Model Loading (Optimized) ---
SHOWUI_MODEL_ID = "showlab/ShowUI-2B"

try:
    # Load ShowUI
    showui_processor = AutoProcessor.from_pretrained(SHOWUI_MODEL_ID, trust_remote_code=True)
    showui_model = Qwen2VLForConditionalGeneration.from_pretrained(
        SHOWUI_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# --- Image Preprocessing ---
def preprocess_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return None

# --- Inference Function ---
def inference(image, query):
    if image is None:
        logger.error("Image preprocessing failed")
        return {"error": "Image preprocessing failed"}

    try:
        # --- UI Grounding Prompt ---
        _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
        min_pixels = 256 * 28 * 28
        max_pixels = 1344 * 28 * 28
        dummy_image_name = "image.png"  # Placeholder filename

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM},
                    {"type": "image", "image": dummy_image_name, "min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": query},  # User's query about location
                ],
            }
        ]

        if torch.cuda.is_available() and showui_model.device.type != 'cuda':
            showui_model.to("cuda")

        # Process text and images separately
        text_prompt = showui_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = showui_processor(text=text_prompt, return_tensors="pt")
        image_inputs = showui_processor(images=image, return_tensors="pt")["pixel_values"]

        # Combine inputs
        inputs["pixel_values"] = image_inputs

        if torch.cuda.is_available():
            inputs = inputs.to("cuda", torch.bfloat16)

        generated_ids = showui_model.generate(**inputs, max_new_tokens=128)

        output_text = showui_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract coordinates using ast.literal_eval
        try:
            click_xy = ast.literal_eval(output_text)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Could not extract coordinates from output: {output_text}, Error: {e}")
            return {"error": f"Could not extract coordinates from output: {output_text}, Error: {e}"}

        if not isinstance(click_xy, list) or len(click_xy) != 2:
            logger.error(f"Invalid coordinate format: {click_xy}")
            return {"error": f"Invalid coordinate format: {click_xy}"}

        result = {
            "coordinates": click_xy,
            "debug_output": output_text,
        }

        logger.info(f"Inference result: {result}")

        return result

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": f"Inference error: {e}"}

# --- RunPod Handler ---
def handler(event):
    try:
        input_data = event['input']
        image_base64 = input_data.get('image')
        query = input_data.get('query')

        if not image_base64:
            logger.error("No image provided")
            return {"error": "No image provided"}
        if not query:
            logger.error("No query provided")
            return {"error": "No query provided"}

        image = preprocess_image(image_base64)
        result = inference(image, query)
        return result

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": f"Handler error: {e}"}

runpod.serverless.start({"handler": handler})