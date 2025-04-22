from flask import Flask, request, jsonify
from diffusers import StableDiffusionInpaintPipeline
import torch
from utils import convert_base64_to_image, convert_image_to_base64
from PIL import Image

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Your device is: {device}")

# 載入模型
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)



# 啟動 Flask 服務
app = Flask(__name__)

@app.route('/diffusion_inpaint', methods=['POST'])
def diffusion_inpaint():
    """
    Stable Diffusion Inpainting API
    """
    try:
        # 解析輸入 JSON
        input_data = request.get_json()
        if 'img_base64' not in input_data or 'mask_base64' not in input_data:
            return jsonify({"error": "Missing required image data"}), 400

        # 確保圖片格式
        raw_image = convert_base64_to_image(input_data['img_base64']).convert("RGB")
        mask_image = convert_base64_to_image(input_data['mask_base64']).convert("L")
        prompt = input_data.get('prompt', '')

        print(f"Received prompt: {prompt}")

        # 進行 Inpainting
        output = pipe(prompt=prompt, image=raw_image, mask_image=mask_image).images[0]
        output.save("test-1.png")

        # 轉換輸出為 Base64
        output_base64 = convert_image_to_base64(output)
     
        return jsonify({"output_base64": output_base64})

    except Exception as e:
        print(f"API ERROR: {e}")  # 顯示詳細錯誤
        return jsonify({"error": str(e)}), 500

@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify(message="Diffusion Inpainting API is running!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006, threaded=False)
