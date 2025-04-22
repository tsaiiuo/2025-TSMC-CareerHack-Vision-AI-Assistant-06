from flask import Flask, request, jsonify
import pandas as pd
from data_manager import DataManager
from agent.vision_assistant_agent import VisionAssistant
from utils import convert_base64_to_image, convert_image_to_base64

# port 8008
# 圖片會存在 output/00000
# json = {
#     "id": "0000000-0000-0000-0000-000000000000", 這裡注意不要和csv有重複的名字就好
#     "detail": "使用者輸入的問題", 
#     "input": "圖片位置",
# }

# e.g.
# json_test = {
#     "id": "0000000-0000-0000-0000-000000000000",
#     "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated referece image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object",
#     "input": 'ref': './data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/ref.png', 'test': './data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/test.png'}" 
# }

app = Flask(__name__)
app.config['DEBUG'] = True

csv_path = '../data/release_public_set.csv'
data_root = '..'
output_root = '../output'

db = pd.read_csv(csv_path)
db = DataManager(db, data_root)
va = VisionAssistant(debug=True, timeout=50, output_root=output_root, is_thread=True, memory_limit_mb=150)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    ref_b64 = data["input"].get("ref")
    test_b64 = data["input"].get("test")
    
    if ref_b64:
        data["input"]["ref"] = convert_image_to_base64(ref_b64)  # 加入解碼後的 ref

    if test_b64:
        data["input"]["test"] = convert_image_to_base64(test_b64)  # 加入解碼後的 test
        
        
    if not data or "id" not in data:
        return jsonify({"error": "Invalid input data"}), 400
    
    try:
        test_messages, test_artifact, test_ob = db.get_messages(data)
        _id = data["id"]   
        info = pd.Series({"id": data.get("id", _id)})  
        
        va.predict(test_messages, test_artifact, info)
        return jsonify({"Success": "!"})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8008, threaded=False)