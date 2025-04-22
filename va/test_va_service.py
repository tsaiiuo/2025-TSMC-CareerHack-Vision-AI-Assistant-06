import json
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
from PIL import Image
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the module
module_dir = os.path.abspath(os.path.join(current_dir, '../'))

# Add the module directory to sys.path
sys.path.append(module_dir)
import utils



url = "http://34.59.120.186:8009/vision"

def send_request(payload):
    # with open("test_input.json", "r") as file:
    #     payload = json.load(file)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        # response.json()
        print("Response from server:" ,response.json())
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


if __name__ == '__main__':
    num_requests = 1
    csv_path = f'../data/release_public_set.csv'
    data_root = '..'
    output_root = '../output'
    db = pd.read_csv(csv_path)
    ob = db.iloc[15,:]
   
    send_request(get_payload(ob, data_root))
    # # with ThreadPoolExecutor(max_workers=num_requests) as executor:
    # #     futures = [executor.submit(send_request) for _ in range(num_requests)]
    # #     for future in futures:
    # #         future.result()