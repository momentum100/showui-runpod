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

SHOWUI_MODEL_ID = "showlab/ShowUI-2B"

try:
    # Load model and processor
    processor = AutoProcessor.from_pretrained(SHOWUI_MODEL_ID, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        SHOWUI_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    logger.error(f"Error loading model or processor: {e}")
    raise

# Set a chat template before using apply_chat_template()
processor.chat_template = {
    "system": "<|System|>{content}</s>",
    "user": "<|User|>{content}</s>",
    "assistant": "<|Assistant|>{content}</s>"
}

def preprocess_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return None

def inference(image, system_prompt, query, img_url):
    if image is None:
        logger.error("Image preprocessing failed")
        return {"error": "Image preprocessing failed"}

    try:
        min_pixels = 256 * 28 * 28
        max_pixels = 1344 * 28 * 28

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": f'Task: {query}'},
                    {"type": "image", "image": img_url, "min_pixels": min_pixels, "max_pixels": max_pixels},
                ],
            }
        ]

        if torch.cuda.is_available() and model.device.type != 'cuda':
            model.to("cuda")

        # Prepare the chat input using the processor
        text_dict = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # The returned dictionary doesn't include the actual image data, so we add it
        text_dict['images'] = [image]

        inputs = processor(**text_dict, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda", torch.bfloat16)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

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

def handler(event):
    try:
        input_data = event['input']
        image_base64 = input_data.get('image')
        query = input_data.get('query')
        system_prompt = input_data.get('system_prompt', "Based on the screenshot of the page, I give a text description and you give its corresponding location.")
        img_url = "image.png" # Dummy image name referenced in content

        if not image_base64:
            logger.error("No image provided")
            return {"error": "No image provided"}
        if not query:
            logger.error("No query provided")
            return {"error": "No query provided"}

        image = preprocess_image(image_base64)
        result = inference(image, system_prompt, query, img_url)
        return result

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": f"Handler error: {e}"}

runpod.serverless.start({"handler": handler})
