import api
import time
import requests
from utils import convert_image_to_base64
from utils import convert_base64_to_image

def call_google(query: str, **kwargs):
    """ 
    'call_google' returns a summary from searching query on google
    Parameters:
        query (str): The search query to be used in the Google search.
        **kwargs: Additional keyword arguments that can be used for extended functionality
              in future implementations, but are currently unused.
    Returns:
        str: A response message searched by google.
    Example
    -------
        >>> call_google(European Union)
        "Sweden decided to join NATO due to a combin..."
    """
    mock_info = "Sweden decided to join NATO due to a combination of reasons. These include the desire to strengthen its security in light of recent regional developments, notably Russia's aggressive actions, \
                such as its war that has pushed the alliance to expand. Sweden, along with Finland, has been an official partner of NATO since 1994 and has been a major contributor to the alliance, participating in various NATO operations. \
                The Swedish government assessed that joining NATO was the best way to protect its national security. Additionally, Sweden's membership was seen as a means to strengthen NATO itself and add to the stability in the Euro-Atlantic area. \
                The process of joining NATO was completed after overcoming the last hurdles, including obtaining the necessary approvals from all existing member states, such as Turkey and Hungary."
    mock_info = 'Cannot find result, you can search on wikipedia.'
    return mock_info

def call_wikipedia(query: str):
    """ 
    'call_wikipedia' returns a summary from searching query on wiki
    Parameters:
        query (str): The search query to be used in the wiki search.
    Returns:
        str: A response message searched on wiki.
    Example
    -------
        >>> call_wikipedia(European Union)
        "Sweden decided to join NATO due to a combin..."
    """
    mock_info = "Sweden decided to join NATO due to a combination of reasons. These include the desire to strengthen its security in light of recent regional developments, notably Russia's aggressive actions, \
                such as its war that has pushed the alliance to expand. Sweden, along with Finland, has been an official partner of NATO since 1994 and has been a major contributor to the alliance, participating in various NATO operations. \
                The Swedish government assessed that joining NATO was the best way to protect its national security. Additionally, Sweden's membership was seen as a means to strengthen NATO itself and add to the stability in the Euro-Atlantic area. \
                The process of joining NATO was completed after overcoming the last hurdles, including obtaining the necessary approvals from all existing member states, such as Turkey and Hungary."

    return mock_info


def calculate(expression: str):
    """
    'calculate' is a toolEvaluates a mathematical expression provided as a string and returns the result.
    The expression is evaluated using Python's built-in eval function, so it should be
    formatted using Python's syntax for mathematical operations.
    Parameters:
        expression (str): The mathematical expression to evaluate. This should be a valid 
                        Python expression using floating point syntax if necessary.
    Returns:
        str: The result of the evaluated expression converted to a string.
    Example:
    -------
        >>> calculate("4 * 7 / 3")
        '9.333333333333334'
    """
    return str(eval(expression))

def call_grounding_dino(prompt, image):
    """'call_grounding_dino' is a tool that can detect and count multiple objects given a text
    prompt such as category names or referring expressions. The categories in text prompt
    are separated by commas or periods. It returns a list of bounding boxes with
    normalized coordinates, label names and associated probability scores.

    Parameters:
        prompt (str): The prompt to ground to the image.
        image (PIL.Image): The image to ground the prompt to.

    Returns:
        List[List[float]]: A list containing the bounding box of the detected objects (xmin, ymin, xmax, ymax). 
            xmin and ymin are the coordinates of the top-left and xmax and ymax are 
            the coordinates of the bottom-right of the bounding box.

    Example:
    -------
        >>> call_grounding_dino("cat", image)
        [[1353, 183, 1932., 736],
        [296, 64, 1147, 663]]
    """
    url = "http://34.59.120.186:8001/dino"
    payload = {
        'text': prompt,
        'img_base64': convert_image_to_base64(image)
    }

    headers = {'Content-Type': 'application/json'}
    tic = time.time()
    response = requests.post(url, json=payload, headers=headers)
    toc = time.time()
    print(f"DINO spend: {round(toc-tic, 3)} s")
    bboxes = None
    if response.status_code == 200:
        print("Response from server:", response.json())
        bboxes = response.json()
    else:
        print("Failed to connect to the server")
    return bboxes

def call_sam(bboxes, image):
    """'call_sam' is a tool that produces high quality object masks 
        from input prompts such as points or boxes, and it can be used to 
        generate masks for all objects in an image. It return a PIL image 
        where the areas containing target objects are labeled as 255, and all 
        other areas are labeled as 0. (The input prompt is used to indicate 
        potential areas where a segment target may be present.)

    Parameters:
        bboxes (List[List[List[float], List[float]]]): This is a list containing multiple bounding boxes, where each bounding box is represented as [[xmin,ymin],[xmax,ymax]].
            xmin and ymin are the coordinates of the top-left and xmax and ymax are the coordinates of the bottom-right of the bounding box.
        image (PIL.Image): The image to ground the input prompt bboxes to.

    Returns:
        PIL.Image: The image where the areas containing target objects are labeled as 255, and all 
        other areas are labeled as 0.

    Example:
    -------
        >>> call_sam([[[50, 30], [150, 130]], [[200, 100], [300, 250]]], image)
        PIL.Image
    """
    headers = {'Content-Type': 'application/json'}
    url = "http://34.59.120.186:8002/sam"
    bboxes_for_sam = []
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        bboxes_for_sam.append([[x1,y1], [x2,y2]])
    print([bboxes_for_sam])
    payload = {
        'bboxes': [bboxes_for_sam],
        'img_base64': convert_image_to_base64(image)
    }
    tic = time.time()
    response = requests.post(url, json=payload, headers=headers)
    toc = time.time()
    print(f"SAM spend: {round(toc-tic, 3)} s")
    if response.status_code == 200:
        print("Response from server:", response.json())
        mask = response.json()
        return convert_base64_to_image(mask['mask'])
    else:
        print("Failed to connect to the server")
    
