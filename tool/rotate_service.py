import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from flask import Flask, request, jsonify
from utils import convert_base64_to_image, convert_image_to_base64  # 新增的導入

# ----------------- 原有处理函数 ---------------------
def extract_foreground(generated_image, mask):
    """
    Extracts the foreground object from the generated image using the provided mask.
    The mask should be a binary image where the object (e.g., dog) is 255 and the background is 0.
    The function returns a new image with an alpha channel, where only the object is visible and the background is transparent.

    Parameters:
        generated_image (PIL.Image): The generated image that includes both the object and its background.
        mask (PIL.Image): A binary mask image with the object marked as 255 and background as 0.

    Returns:
        PIL.Image: An RGBA image where the alpha channel is derived from the mask, effectively isolating the object.
    
    Example:
    --------
        >>> fg_image = extract_foreground(generated_image, mask)
        >>> fg_image.show()
    """
    # Convert the generated image to RGBA to support transparency
    image_rgba = generated_image.convert("RGBA")
    
    # Convert the mask to grayscale (if not already) and ensure it's binary
    mask = mask.convert("L")
    # Optionally threshold the mask (if not already binary)
    threshold = 127
    binary_mask = mask.point(lambda p: 255 if p > threshold else 0)
    
    # Set the alpha channel of the RGBA image to be the binary mask
    image_rgba.putalpha(binary_mask)
    
    return image_rgba
def remove_mask_from_image(image, mask):
    """
    Removes the masked region from the input image by inpainting.

    Parameters:
        image (PIL.Image): The input image.
        mask (PIL.Image): The mask image where regions to be removed are marked as 255 and other areas as 0.

    Returns:
        PIL.Image: The inpainted image with the masked regions removed.

    Example:
    --------
        >>> result = remove_mask_from_image(input_image, mask_image)
    """
    # Convert the input PIL image to a NumPy array
    image_np = np.array(image)
    
    # Convert the mask to a NumPy array and ensure it's uint8
    mask_np = np.array(mask).astype(np.uint8)
    
    # Ensure mask is single channel (grayscale)
    if len(mask_np.shape) == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    
    # Optionally, threshold the mask to ensure binary values (0 or 255)
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    # Use OpenCV's inpainting to remove the masked region
    inpaint_radius = 3  # Adjust as needed
    inpainted_np = cv2.inpaint(image_np, mask_np, inpaint_radius, cv2.INPAINT_TELEA)
    
    # Convert the inpainted NumPy array back to a PIL.Image
    inpainted_image = Image.fromarray(inpainted_np)
    return inpainted_image
def add_generated_image_to_region(ref_image, dog_image, boxes):
    """
    Places the generated dog image into the region specified by the bounding box on the reference image.
    
    Parameters:
        ref_image (PIL.Image): The reference image with the original object removed (inpainted background).
        dog_image (PIL.Image): The generated dog image.
        boxes (List[List[float]]): A list containing one or more bounding boxes in the format 
                                   [xmin, ymin, xmax, ymax]. Coordinates are expected to be in pixel units.
    
    Returns:
        PIL.Image: The composited image with the dog image inserted into the region defined by the first bounding box.
    
    Example:
    --------
        >>> final_image = add_generated_image_to_region(ref_image, dog_image, [[50, 30, 150, 130]])
    """
    if not boxes:
        print("No bounding boxes provided; returning the original image.")
        return ref_image

    # Use the first bounding box
    box = boxes[0]
    xmin, ymin, xmax, ymax = map(int, box)
    target_width = xmax - xmin
    target_height = ymax - ymin

    # Resize the dog image to fit within the bounding box
    dog_resized = dog_image.resize((target_width, target_height))

    # Create a copy of the reference image
    composite_image = ref_image.copy()

    # Paste the resized dog image onto the reference image; if the image has transparency, use it as mask.
    composite_image.paste(dog_resized, (xmin, ymin), dog_resized if dog_resized.mode == 'RGBA' else None)
    return composite_image
def detect_and_match_features(ref, test):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref, None)
    kp2, des2 = sift.detectAndCompute(test, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return src_pts, dst_pts

def estimate_rotation_translation(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask

def rotate_image(image, M):
    h, w = image.shape[:2]
    rotated_image = cv2.warpPerspective(image, M, (w, h))
    return rotated_image

def crop_black_padding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y+h, x:x+w]
        return cropped, (x, y, w, h)
    else:
        return image, (0, 0, image.shape[1], image.shape[0])

def remove_noise(image):
    denoised_image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return denoised_image

def calculate_residual(ref, test):
    residual = np.abs(ref.astype(np.int16) - test.astype(np.int16)).astype(np.uint8)
    return residual

def process_residual(residual):
    # 将四周30像素置零
    residual[:20, :] = 0
    residual[-20:, :] = 0
    residual[:, :20] = 0
    residual[:, -20:] = 0
    _, binary_mask = cv2.threshold(residual, 30, 255, cv2.THRESH_BINARY)
    binary_mask[binary_mask != 0] = 255
    binary_mask_bool = binary_mask.astype(bool)
    cleaned_mask = remove_small_objects(binary_mask_bool, min_size=64, connectivity=2)
    final_mask = cleaned_mask.astype(np.uint8) * 255
    if final_mask.ndim > 2:
        final_mask = np.squeeze(final_mask)
    return Image.fromarray(final_mask, mode='L')

def compute_iou(mask1, mask2):
    arr1 = np.array(mask1)
    arr2 = np.array(mask2)
    arr1_bool = arr1 > 0
    arr2_bool = arr2 > 0
    intersection = np.logical_and(arr1_bool, arr2_bool).sum()
    union = np.logical_or(arr1_bool, arr2_bool).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

# def apply_local_clahe(gray_img, block_size=(64, 64), contrast_threshold=20, clahe_clipLimit=2.0, tileGridSize=(8,8)):
#     H, W = gray_img.shape
#     enhanced_img = np.zeros_like(gray_img)
#     clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=tileGridSize)
#     for i in range(0, H, block_size[0]):
#         for j in range(0, W, block_size[1]):
#             block = gray_img[i:min(i+block_size[0], H), j:min(j+block_size[1], W)]
#             contrast = np.max(block) - np.min(block)
#             if contrast < contrast_threshold:
#                 block_enhanced = clahe.apply(block)
#                 enhanced_img[i:min(i+block.shape[0], H), j:min(j+block.shape[1], W)] = block_enhanced
#             else:
#                 enhanced_img[i:min(i+block.shape[0], H), j:min(j+block.shape[1], W)] = block
#     return enhanced_img

# ----------------- 微服务部分 ---------------------

app = Flask(__name__)
@app.route('/rotate', methods=['POST'])
def rotate_api():
    """
    Receives a POST request with a JSON payload containing two base64 encoded images:
        "ref": Base64 string of the reference image.
        "test": Base64 string of the test image.
    This API aligns the reference image using the test image by performing rotation,
    and returns the rotated image as a base64 string.
    """
    data = request.get_json()
    ref_b64 = data.get("ref")
    test_b64 = data.get("test")
    
    if not ref_b64 or not test_b64:
        return jsonify({"error": "Both 'ref' and 'test' base64 image strings must be provided."}), 400

    try:
        ref_img = convert_base64_to_image(ref_b64)
        test_img = convert_base64_to_image(test_b64)
    except Exception as e:
        return jsonify({"error": "Image decoding error: " + str(e)}), 400

    ref_np = np.array(ref_img)
    test_np = np.array(test_img)
    
    # Convert to grayscale to extract features
    ref_gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test_np, cv2.COLOR_RGB2GRAY)
    src_pts, dst_pts = detect_and_match_features(ref_gray, test_gray)
    M, _ = estimate_rotation_translation(src_pts, dst_pts)
    
    rotated_ref = rotate_image(ref_np, M)
    rotated_img = Image.fromarray(rotated_ref)
    return jsonify({"rotated": convert_image_to_base64(rotated_img)})

# ----------------------- 2. Denoise API -----------------------
@app.route('/denoise', methods=['POST'])
def denoise_api():
    """
    Receives a POST request with a JSON payload containing:
        "image": Base64 string of the image.
    This API applies noise removal to the input image and returns the denoised image as a base64 string.
    """
    data = request.get_json()
    image_b64 = data.get("image")
    
    if not image_b64:
        return jsonify({"error": "The 'image' base64 string must be provided."}), 400

    try:
        img = convert_base64_to_image(image_b64)
    except Exception as e:
        return jsonify({"error": "Image decoding error: " + str(e)}), 400

    img_np = np.array(img)
    denoised_np = remove_noise(img_np)
    denoised_img = Image.fromarray(denoised_np)
    return jsonify({"denoised": convert_image_to_base64(denoised_img)})

# ----------------------- 3. Enhance API -----------------------
@app.route('/enhance', methods=['POST'])
def enhance_api():
    """
    Receives a POST request with a JSON payload containing:
        "image": Base64 string of a grayscale image.
    This API applies CLAHE enhancement to the input grayscale image and returns the enhanced image as a base64 string.
    """
    data = request.get_json()
    image_b64 = data.get("image")
    
    if not image_b64:
        return jsonify({"error": "The 'image' base64 string must be provided."}), 400

    try:
        img = convert_base64_to_image(image_b64)
    except Exception as e:
        return jsonify({"error": "Image decoding error: " + str(e)}), 400

    img_np = np.array(img)
    # Convert to grayscale if not already
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    enhanced_np = apply_clahe(img_np)
    enhanced_img = Image.fromarray(enhanced_np)
    return jsonify({"enhanced": convert_image_to_base64(enhanced_img)})
@app.route('/mask', methods=['POST'])
def mask_api():
    """
    Receives a POST request with a JSON payload containing two base64 encoded images:
        "image1": Base64 string of the first enhanced image.
        "image2": Base64 string of the second enhanced image.
    This API calculates the residual (difference) between the two images and then processes
    the residual image to generate the final binary mask (defects as 255, background as 0).
    The final mask is returned as a base64 string.
    """
    data = request.get_json()
    image1_b64 = data.get("image1")
    image2_b64 = data.get("image2")
    
    if not image1_b64 or not image2_b64:
        return jsonify({"error": "Both 'image1' and 'image2' base64 strings must be provided."}), 400

    try:
        img1 = convert_base64_to_image(image1_b64)
        img2 = convert_base64_to_image(image2_b64)
    except Exception as e:
        return jsonify({"error": "Image decoding error: " + str(e)}), 400

    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Calculate the residual image (difference)
    residual_np = calculate_residual(img1_np, img2_np)
    
    # Optionally, you can convert the residual to a PIL Image for debugging:
    # residual_img = Image.fromarray(residual_np)
    # residual_img.show()  # Uncomment to display the residual image

    # Process the residual to generate the final binary mask
    mask_img = process_residual(residual_np)  # Returns a PIL.Image object
    
    return jsonify({"mask": convert_image_to_base64(mask_img)})

@app.route('/sift_bruteforce_rotate_residual_mask_diff', methods=['POST'])
def diff():
    """
    接收 POST 请求，输入 JSON 数据中包含三张图片的 base64 编码：
        "ref": 参考图 (ref) 的 base64 字符串
        "test": 测试图 (test) 的 base64 字符串
    该服务会执行对齐、去噪、CLAHE增强、残差计算和 mask 生成流程，
    生成 mask 并返回一个JSON，
    """
    data = request.get_json()
    ref_b64 = data.get("ref")
    test_b64 = data.get("test")

    
    if not ref_b64 or not test_b64 :
        return jsonify({"error": "需要提供 ref, test, 和 defect 三张图片的 base64 字符串"}), 400

    try:
        # 利用 convert_base64_to_image 将 base64 字符串转换为 PIL Image 对象
        ref_img = convert_base64_to_image(ref_b64)
        test_img = convert_base64_to_image(test_b64)
    except Exception as e:
        return jsonify({"error": "解码图片出错: " + str(e)}), 400

    # 转换为 numpy 数组，后续使用 OpenCV 处理
    ref = np.array(ref_img)
    test = np.array(test_img)
    
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    
    src_pts, dst_pts = detect_and_match_features(ref_gray, test_gray)
    M, _ = estimate_rotation_translation(src_pts, dst_pts)
    
    # 旋转参考图像使其与 test 对齐
    rotated_ref = rotate_image(ref, M)
    
    denoised_rotated_ref = remove_noise(rotated_ref)
    denoised_test = remove_noise(test)
    
    denoised_rotated_ref_gray = cv2.cvtColor(denoised_rotated_ref, cv2.COLOR_RGB2GRAY)
    denoised_test_gray = cv2.cvtColor(denoised_test, cv2.COLOR_RGB2GRAY)
    
    enhanced_ref_gray = apply_clahe(denoised_rotated_ref_gray)
    enhanced_test_gray = apply_clahe(denoised_test_gray)
    
    residual = calculate_residual(enhanced_ref_gray, enhanced_test_gray)
    result = process_residual(residual)
  
    result.save(f"result1.png")
    
    # 利用 convert_image_to_base64 将生成的 mask 转换为 base64 字符串
    result_b64 = convert_image_to_base64(result)
    
    
    return jsonify({"mask": result_b64})


@app.route('/replace_mask_by_generated_image', methods=['POST'])
def replace_api():
    """
    Receives a POST request with a JSON payload containing:
        "ref_image": Base64 string of the reference image with the object removed.
        "generated_image": Base64 string of the generated image.
        "boxes": A list of bounding boxes where each box is [xmin, ymin, xmax, ymax].
    This API composites the dog image into the removed region of the reference image 
    according to the provided bounding box and returns the final composited image as a base64 string.
    """
    data = request.get_json()
    if not data or "ref_image" not in data or "generated_image" not in data or "boxes" not in data:
        return jsonify({"error": "Missing required parameters: ref_image, generated_image, and boxes."}), 400

    try:
        ref_img = convert_base64_to_image(data["ref_image"])
        dog_img = convert_base64_to_image(data["generated_image"])
        boxes = data["boxes"]
        
        final_image = add_generated_image_to_region(ref_img, dog_img, boxes)
        final_image.save("replace_generated_image.png")
        final_b64 = convert_image_to_base64(final_image)
        return jsonify({"final_image": final_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/remove_mask', methods=['POST'])
def remove_mask_api():
    """
    Receives a POST request with a JSON payload containing:
        "image": Base64 string of the input image.
        "mask": Base64 string of the mask image (masked regions as 255).
    
    This API applies inpainting to remove the masked region from the image
    and returns the resulting image as a base64-encoded string.
    """
    data = request.get_json()
    if not data or "image" not in data or "mask" not in data:
        return jsonify({"error": "Both 'image' and 'mask' must be provided."}), 400

    try:
        # Use helper functions to decode the base64 strings into PIL.Image objects.
        image = convert_base64_to_image(data["image"])
        mask = convert_base64_to_image(data["mask"])
        
        # Process the image with the inpainting function.
        result_image = remove_mask_from_image(image, mask)
        # Convert the resulting image back to a base64 string using your helper.
        result_image.save("remove_mask_image.png")
        result_b64 = convert_image_to_base64(result_image)
        
        return jsonify({"result_image": result_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/extract_foreground', methods=['POST'])
def extract_foreground_api():
    """
    Receives a POST request with a JSON payload containing:
        "generated_image": Base64 string of the generated image (including background).
        "mask": Base64 string of the binary mask (object marked as 255, background as 0).
    This API extracts the foreground object by applying the mask as the alpha channel
    and returns the resulting image as a base64-encoded string.
    """
    data = request.get_json()
    if not data or "generated_image" not in data or "mask" not in data:
        return jsonify({"error": "Both 'generated_image' and 'mask' must be provided."}), 400

    try:
        # Decode the images using the helper functions.
        gen_img = convert_base64_to_image(data["generated_image"])
        mask_img = convert_base64_to_image(data["mask"])
    except Exception as e:
        return jsonify({"error": "Image decoding error: " + str(e)}), 400

    try:
        # Extract the foreground using the defined function.
        fg_image = extract_foreground(gen_img, mask_img)
        fg_image.save("remove_fg_image.png")
        fg_b64 = convert_image_to_base64(fg_image)
        return jsonify({"foreground": fg_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, threaded=False)
