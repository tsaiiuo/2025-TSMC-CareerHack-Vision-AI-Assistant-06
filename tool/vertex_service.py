import time
from flask import Flask, request, jsonify
from PIL import Image
import vertexai
import numpy as np
from vertexai.preview.vision_models import ImageGenerationModel
from utils import convert_base64_to_image, convert_image_to_base64  # Make sure this function converts a PIL.Image to a base64 string
# Initialize your Vertex AI project and model.
PROJECT_ID = "tsmccareerhack2025-aaid-grp6"
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
output_file = "input-image.png"

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    Receives a POST request with a JSON payload containing:
        "prompt": A text prompt describing what you want to see.
        (Optional) "number_of_images": Number of images to generate (default: 1).
        (Optional) "aspect_ratio": Desired aspect ratio (default: "1:1").
        (Optional) "language": Language for the prompt (default: "en").
        (Optional) "safety_filter_level": Safety filter level (default: "block_some").
        (Optional) "person_generation": Person generation parameter (default: "allow_adult").
    This API uses Vertex AI's ImageGenerationModel to generate images based on the prompt,
    and returns the first generated image as a base64-encoded string along with the size of the image bytes.
    """
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "A 'prompt' must be provided."}), 400

    prompt = data.get("prompt")
    number_of_images = data.get("number_of_images", 1)
    aspect_ratio = data.get("aspect_ratio", "1:1")
    language = data.get("language", "en")
    safety_filter_level = data.get("safety_filter_level", "block_some")
    person_generation = data.get("person_generation", "allow_adult")
    print(f"test")
    try:
        tic = time.time()
        images = model.generate_images(
            prompt=prompt,
            number_of_images=number_of_images,
            language="en",
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation=person_generation,
        )
        toc = time.time()
        print(f"Image generation took: {round(toc-tic, 3)} seconds")
        
        images[0].save(location=output_file)
        # Convert the PIL.Image to a base64 string using the helper function
        img_path = f"./input-image.png"
        generated_img = Image.open(img_path)
        
        img_b64 = convert_image_to_base64(generated_img)
        return jsonify({
            "generated_image": img_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8004, threaded=False)