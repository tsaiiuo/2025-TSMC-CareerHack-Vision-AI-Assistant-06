import os
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
from flask_cors import CORS  # Import CORS
import requests
# Construct the path to the module
module_dir = os.path.abspath(os.path.join(current_dir, '../'))

# Add the module directory to sys.path
sys.path.append(module_dir)
import utils
from flask import Flask, jsonify, request
import json
import time
import pandas as pd
from PIL import Image
from utils import check_dir, convert_image_to_base64, parser_np
from data_manager import DataManagerService
from agent.vision_assistant_agent import VisionAssistant

url = "http://34.59.120.186:8009/vision"

def send_request(payload):
    # with open("test_input.json", "r") as file:
    #     payload = json.load(file)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        # response.json()
        print("Response from server:",response.json() )
        return response.json()
    else:
        print("Failed to connect to the server")

def get_payload(ob, data_root):
    payload = {
        "groupToken": "XXX",
        "uuid": ob['id'],
        "messages": ob['messages'],
        "detail": ob['detail'],
        "outputType": ob['output_format'],
        "artifacts": []
    }
    payload["artifacts"] = get_artifact(ob, data_root)
    return payload

 # utils.convert_image_to_base64(Image.open(os.path.join(data_root, input_dict[k]))),
def get_artifact(ob, data_root):
    ret = []
    input_dict = eval(ob['input'])
    for k in input_dict:
        obj = {
            "base64": utils.convert_image_to_base64(Image.open(os.path.join(data_root, input_dict[k]))),
            "type": 'png',
            "name": k
        }
        ret.append(obj)
    # print(ret)
    return ret


app = Flask(__name__)
CORS(app)
@app.route('/vision', methods=['POST'])
def process_data():
    input_data = request.get_json()
    image_data = input_data["artifacts"]
    

    prompt = input_data["messages"]
   
    detail_prompt = input_data["detail"]

    output_type = input_data["outputType"]
    uuid = input_data["uuid"]

    df = pd.DataFrame({"messages": [prompt], "detail": [detail_prompt], "input": [json.dumps(image_data)]})
    db = DataManagerService(df)
    messages, images, ob = db[0]
    # print(messages[0])
    
    va = VisionAssistant(debug=False, timeout=120, memory_limit_mb=200)
    
    result = va.predict(messages, images)
    if isinstance(result, Image.Image):
        result = convert_image_to_base64(result)
    elif isinstance(result, (int, float)):
        result = float(result)
    
    result = parser_np({'result': result})
    
    record=va.get_thinking_log()
    return jsonify({
        "result": result['result'],
        "record":record,
        "type": output_type,
        "uuid": uuid
    })


@app.route('/test', methods=['POST'])
def test_data():
    input_data = request.get_json()
    num_requests = 1
    print("Current working directory:", os.getcwd())

    csv_path = f'../data/release_public_set.csv'
    data_root = '..'
    output_root = '../output'
    db = pd.read_csv(csv_path)
    
    ob = db.iloc[input_data.get("id"),:]
   
    result=send_request(get_payload(ob, data_root))
    print(result)
    
    chain = result['record']
    modified_text = chain.replace("\n", "*")

    # Split the chain text by newline
    chain_array = chain.split("\n")
    return jsonify({
        "result": result['result'],
        "record_string":modified_text,
        "record":chain_array,
    })



@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009, threaded=True)