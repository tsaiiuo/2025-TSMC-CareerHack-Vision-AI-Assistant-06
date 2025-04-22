import pandas as pd
from data_manager import DataManager
from agent.vision_assistant_agent import VisionAssistant
import time
from utils import check_dir
import shutil
from PIL import Image
from utils import convert_base64_to_image, convert_image_to_base64


test = {
    "id": "bd27cf15-c7fe-44b7-a7e7-9578f1a21088",
    "messages": "Adjust the orientation of the refence image using the test image. Display where the difference. Supply the end result as a PIL.Image object",
    "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated referece image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object",
    "outputType":"base64, PIL uint8",
    "input": "{'ref': './data/public_set/bd27cf15-c7fe-44b7-a7e7-9578f1a21088/ref.png', 'test': './data/public_set/bd27cf15-c7fe-44b7-a7e7-9578f1a21088/test.png'}",
    "artifacts":[
        {
            "base64": "",
            "type": "",
            "name": ""
        }
    ]
}

density_test = {
    "id": "867d3144-ffcf-45a4-8085-5c10a7a5fe9c",
    "detail": "Count the pattern on the images.  The picture area is 73 square meters. Calculate the density (count/per square meters) of the pattern. (Give the final answer in float or int using result variable. The graylevel of pattern is larger than background. Please use image processing method (such as otsu) to get the binary mask.)",
    "outputType":"base64, PIL uint8",
    "input": "{'ref': './data/public_set/867d3144-ffcf-45a4-8085-5c10a7a5fe9c/ref.png'}"
}




user_test = {
    "id": "0000000-0000-0000-0000-000000000000",
    "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated referece image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object"
}

# user_test = {
#     "id": "0000000-0000-0000-0000-000000000000",
#     "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated referece image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object",
#     "input": "{'ref': './data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/ref.png', 'test': './data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/test.png'}" 
# }

if __name__ == '__main__':
    csv_path = f'../data/release_public_set.csv'
    data_root = '..'
    output_root = '../output'

    db = pd.read_csv(csv_path)
    db = DataManager(db, data_root)
    # db = DataManager(db, data_root)
    va = VisionAssistant(debug=True, timeout=50, output_root=output_root, is_thread = True, memory_limit_mb = 150)
    tic = time.time()
    input_paths = None
    obj_id = None
    
    user_test = {
    "id": "0000000-0000-0000-0000-000000000000",
    "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated referece image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object"
}
    test_json = user_test
    
    ref_ = "../data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/ref.png"
    test_ = "../data/public_set/04f255ed-5824-4853-96e1-be41d14eaa4c/test.png"
    
    ref_img = Image.open(ref_)
    test_img = Image.open(test_)

    user_test = {
        "id": "0000000-0000-0000-0000-000000000000",
        "detail": "Detect SIFT features in both the reference and test images. Match features using a Brute Force matcher. Estimate the rotation angle and translation using RANSAC. Rotate the reference image using the estimated homography matrix. Remove the noise of rotated reference image and test image. Calculate the residual using gray images and set 30 pixel border of residual to 0. Display where the outlier of residual in binary mask and remove small objects. Supply the end result as a PIL.Image object.",
        "input": {
            "ref": ref_img,
            "test": test_img
        }
    }
    
    ref_img = user_test["input"].get("ref")
    test_img = user_test["input"].get("test")
    
    # user_test["input"]["ref"] = convert_image_to_base64(ref_img)  # 加入解碼後的 ref
    # user_test["input"]["test"] = convert_image_to_base64(test_img)  # 加入解碼後的 tes
    

    if ref_img:
        user_test["input"]["ref"] = convert_image_to_base64(ref_img)  # 加入解碼後的 ref

    if test_img:
        user_test["input"]["test"] = convert_image_to_base64(test_img)  # 加入解碼後的 test
    
    
    test_messages, test_artifact, test_ob = db.get_messages(user_test)
    _id = user_test["id"]
    
    info = pd.Series({"id": test_json.get("id", _id)})  
    va.predict(test_messages, test_artifact, info)  


    
    
#     for messages, artifact, ob in db:
#         if ob['id'] == "bd27cf15-c7fe-44b7-a7e7-9578f1a21088":
#             # result = va.predict(messages, artifact)
#             input_paths = artifact
#             obj_id = ob['id']
#             va.add_task(messages, artifact, ob)
    
#     va.start_task(1)
#     toc=  time.time()
#     print(f'Done in {round(toc-tic, 3)} sec.')
#     va.dump_record()
    # output_path = f"../output/{obj_id}"
    # for input_path in input_paths:
    #     name = input_path.split('/')[-1].split('.')[0]
    #     shutil.copy(input_path, f'{output_path}/{name}.png')
