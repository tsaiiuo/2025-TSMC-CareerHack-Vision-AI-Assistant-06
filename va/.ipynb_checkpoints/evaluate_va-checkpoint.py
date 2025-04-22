import os
from flask import jsonify
import requests
import pandas as pd
import base64
import concurrent.futures
import logging
import utils
import sys
from PIL import Image

logger = logging.getLogger(__name__)

# # Get the directory of the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to the module
# module_dir = os.path.abspath(os.path.join(current_dir, '../'))

# # Add the module directory to sys.path
# sys.path.append(module_dir)


from evaluation.evaluation import create_radar_chart, compile_task_scores
from dataclasses import dataclass

@dataclass
class Config:
    EVAL_MAX_WORKER_PER_USER = 5
    EVAL_PRIVATE_TA_SOLUTION_RELATIVE_PATH = "../data/private_service_set(TA).csv"
    EVAL_PUBLIC_TA_SOLUTION_RELATIVE_PATH = "../data/release_public_set.csv"


def fetch_test_case_info(uuid, ta_solution_df, data_root):
    task_dict_list = ta_solution_df[ta_solution_df['id'] == uuid].to_dict("records")
    task_dict = task_dict_list[0]

    # task_dict['messages'] = task_dict_list['messages']
    # task_dict['detail'] = task_dict_list['detail']
    input_info = eval(task_dict['input'])
    ret = []
    for k in input_info:
        img_path = os.path.join(data_root, input_info[k])
        obj = {
            "base64": utils.convert_image_to_base64(Image.open(img_path)),
            "type": 'png',
            "name": k
        }
        ret.append(obj)
    # print(task_dict)
    task_dict['artifacts'] = ret
    return task_dict

def get_title(eval_request_payload: dict):
    va_token = eval_request_payload.get('groupToken')
    return f"{va_token}_radar_chart"

def get_request_count(group_token: str):
    # haven't implemented yet; return 1 for now
    return 1

def format_score_report(score_report: pd.DataFrame, dataset_type: str, group_token: str):
    final_score_report = dict()

    editted_score_report = score_report[['id', 'score']]
    editted_score_report = editted_score_report.rename(columns={'id': 'uuid'})
    final_score_report["result"] = editted_score_report.to_dict(orient='records')

    final_score_report["query time"] = get_request_count(group_token)

    return jsonify(final_score_report)

# Function to call the API and store the response
def call_va_service(url, token, uuid, ta_solution_df, data_root):
    test_case_info_dict = fetch_test_case_info(uuid, ta_solution_df, data_root)
    payload = {
        "uuid": uuid,
        "groupToken": token,
        "messages": test_case_info_dict["messages"],
        "detail": test_case_info_dict["detail"],
        "outputType": test_case_info_dict["output_format"],
    }
    print(payload)
    payload['artifacts'] = test_case_info_dict["artifacts"]

    try:
        headers = {'Content-Type': 'application/json'}
        print(url)
        # [print(payload[i]) for i in payload if i !='artifacts']
        # response = requests.post(url, json=payload, headers=headers, timeout=200)

        from urllib3.util import Retry
        from requests.adapters import HTTPAdapter
        # define the retry strategy
        retry_strategy = Retry(
            total=3,  # maximum number of retries
        )
        # create an HTTP adapter with the retry strategy and mount it to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)

        # create a new session object
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # make a request using the session object
        response = session.post(url, json=payload, headers=headers, timeout=200)

        if response.status_code == 200:
            print("Response from server:", response.json())
            return response.status_code, response.json()
        else:
            print(response.text)
            print("Failed to connect to the server")
            logger.error(f"Bad VA service response;")
            logger.error(f"VA service Response Code: {response.status_code}")
            logger.error(f"VA service Response Message: {response.text}")
        return response.status_code, response.text
    except Exception as e:
        print(e)
        # print(f"Error loading the VA service response;")
        # print(f"response: {response.text}")
        logger.error(f"Unexpected error: {e}")
        logger.error(f"VA service Response Code: {response.status_code}")
        logger.error(f"VA service Response Message: {response.text}")
        return response.status_code, response.text

def evaluate(data, output_path, data_root):
    va_url = data.get('visionAssistantURL')
    va_token = data.get('groupToken')
    uuids = data.get('dataList')
    dataset_type = data.get('dataset')  # can only be either private or public
    max_workers = Config.EVAL_MAX_WORKER_PER_USER  # Default to 5 workers if not specified
    if not va_url or not va_token or not dataset_type:
       	return jsonify({"error": "Missing required parameters"}), 400

    # uuids can be None; when dataList is not provided uuids will be None and all the test cases
    # of the selected dataset_type (private or public) will be ran
    if uuids is not None:
        if not isinstance(uuids, list):
            return jsonify({"error": "dataList needs to be a list"}), 400

    # load ta_solution dataframe
    if dataset_type == 'private':
        ta_solution_df = pd.read_csv(Config.EVAL_PRIVATE_TA_SOLUTION_RELATIVE_PATH)
    elif dataset_type == 'public':
        ta_solution_df = pd.read_csv(Config.EVAL_PUBLIC_TA_SOLUTION_RELATIVE_PATH)
    else:
        return jsonify({"error": "dataset needs to be either private or public"})
        
    if uuids:
        # screen out the test cases that were not submitted by the user
        ta_solution_df = ta_solution_df[ta_solution_df['id'].isin(uuids)]
    else:
        # if dataList is not provided (hence uuids will be None), all the test cases should be ran.
        # set uuids to be all the test cases' id in ta_solution_df for either private or public depending on the dataset_type
        uuids = ta_solution_df['id'].unique().tolist()

    # screen out the test cases that were not submitted by the user
    ta_solution_df = ta_solution_df[ta_solution_df['id'].isin(uuids)]

    results = []
    for obj_id in uuids:
        results.append({"uuid": obj_id, "result": "Fail", "type": "N/A"})

    # Create a ThreadPoolExecutor with a maximum number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary to map futures to their UUID index
        future_to_index = {executor.submit(call_va_service, va_url, va_token, uuid, ta_solution_df, data_root): i for i, uuid in enumerate(uuids)}

        for future in concurrent.futures.as_completed(future_to_index, timeout=2000):
            index = future_to_index[future]
            try:
                va_resp_code, va_resp_msg = future.result(timeout=2000)
                if va_resp_code >= 200 and va_resp_code < 300 and isinstance(va_resp_msg, dict):
                    results[index] = va_resp_msg
                else:
                    print('result error')
                    pass
            except Exception as e:
                logger.error(f"{e}")

    # for i, uuid in enumerate(uuids):
    #     results[i] = call_va_service(va_url, va_token, uuid, ta_solution_df)

    results_df = pd.DataFrame(results)
    # print(f"{results}")
    # to align current version of results, which does not include the type column
    results_df = results_df[["uuid", "result"]]
    results_df = results_df.rename(columns={"uuid": "id"})
    results_df.to_csv(os.path.join(output_path, 'result.csv'))
    return results_df

def get_payload(va_url, dataList):
    payload = {
        "groupToken": "XXXX",
        "visionAssistantURL": va_url,
        "dataset": "public",
        "dataList": dataList,
    }
    print(payload)
    return payload

if __name__ == '__main__':
    va_url = "http://192.168.1.100:8003/vision"
    dataList = None
    # dataList = [
    #     "b377a2d3-bbec-4a0a-9a41-67513bf8e885"
    #     ]
    output_path = '../output'
    data_root = '../'
    data = get_payload(va_url, dataList)
    evaluate(data, output_path, data_root)