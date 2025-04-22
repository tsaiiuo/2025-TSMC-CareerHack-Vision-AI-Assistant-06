import requests
import time
import json

BASE_URL = "http://127.0.0.1:8008/predict"

density_test = {
    "id": "867d3144-ffcf-45a4-8085-5c10a7a5fe9c",
    "detail": "Count the pattern on the images. The picture area is 73 square meters. Calculate the density (count/per square meters) of the pattern. (Give the final answer in float or int using result variable. The graylevel of pattern is larger than background. Please use image processing method (such as otsu) to get the binary mask.)",
    "outputType": "base64, PIL uint8",
    "input": "{'ref': './data/public_set/867d3144-ffcf-45a4-8085-5c10a7a5fe9c/ref.png'}"
}

test = {
    "id": "0000000-0000-0000-0000-000000000000",
    "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated referece image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object",
    "input": "{'ref': './data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/ref.png', 'test': './data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/test.png'}" 
}

def test_predict_api():
    response = requests.post(BASE_URL, json=test)
    # print("Status Code:", response.status_code)
    # print("Response:", response.json())

if __name__ == "__main__":
    test_predict_api()
