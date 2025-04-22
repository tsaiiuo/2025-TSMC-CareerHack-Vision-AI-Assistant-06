from PIL import Image
from flask import Flask, jsonify, request
from utils import convert_base64_to_image, convert_image_to_base64
import numpy as np

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Your device is: {device}")

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

app = Flask(__name__)
@app.route('/sam2', methods=['POST'])
def predict():
    input_data = request.get_json()
    raw_image = input_data['img_base64']
    raw_image = convert_base64_to_image(raw_image)
    bboxes = input_data['bboxes']
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(raw_image)
        masks, _, _ = predictor.predict(box=bboxes, multimask_output=False)
        
#     print(len(masks), '*'*50)
#     print(masks[0].shape, raw_image.size, np.max(masks[0]), masks[0].dtype, '*'*50)
    
#     for i in range(len(masks)):
#         # 将 float32 转换为 uint8（乘以 255，使 1.0 变为 255）
#         array_uint8 = (masks[i][0] * 255).astype(np.uint8)

#         # 转换为 PIL Image（mode='L' 代表灰度图）
#         image = Image.fromarray(array_uint8, mode='L')

#         # 儲存圖片
#         image.save(f"mask{i}.png")

#     print("圖片已儲存為 mask.png")
    h, w = raw_image.size
    print(masks[0].shape, raw_image.size)
    best_masks_for_each_bbox = [np.squeeze(mask) for mask in masks]+[np.zeros((w, h))]
    result_or = np.logical_or.reduce(best_masks_for_each_bbox).astype(np.float32)
    result_image = (result_or * 255).astype(np.uint8)
    result_image = Image.fromarray(result_image, mode='L')
    result_image.save(f"result.png")
    
    return {'mask': convert_image_to_base64(result_image)}

@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, threaded=False)