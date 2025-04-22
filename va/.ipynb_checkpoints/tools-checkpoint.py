import api
import time
import requests
import numpy as np
from utils import convert_image_to_base64, convert_base64_to_image
class FUNCTIONS:
    """ All available functions for the GPT calling """
    """ Format:

        Parameters:
            
        Returns:
            
        Example:
        -------
            >>> 
            
    """

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
            List[List[float]]: A list containing the bounding box of the detected objects [xmin, ymin, xmax, ymax]. 
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

    def call_sam2(bboxes, image):
        """'call_sam2' is a tool that produces high quality object masks 
            from input prompts such bounding boxes, and it can be used to 
            generate one mask for all target objects in an image. It return a PIL.Image 
            where the areas containing target objects are labeled as 255, and all 
            other areas are labeled as 0. (The input prompt is used to indicate 
            potential areas where a segment target may be present.)

        Parameters:
            bboxes (List[List[float]]): This is a list containing multiple bounding boxes, where each bounding box is represented as [xmin,ymin,xmax,ymax].
                xmin and ymin are the coordinates of the top-left and xmax and ymax are the coordinates of the bottom-right of the bounding box.
            image (PIL.Image): The image to ground the input prompt bboxes to.

        Returns:
            PIL.Image: The image where the areas containing target objects are labeled as 255, and all 
            other areas are labeled as 0.

        Example:
        -------
            >>> call_sam2([[50, 30, 150, 130]], image)
            PIL.Image
        """
        headers = {'Content-Type': 'application/json'}
        url = "http://34.59.120.186:8002/sam2"
        payload = {
            'bboxes': bboxes,
            'img_base64': convert_image_to_base64(image)
        }
        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"SAM spend: {round(toc-tic, 3)} s")
        if response.status_code == 200:
            print("Response from server:", response.json())
            output = response.json()
            mask_image = convert_base64_to_image(output['mask'])
            
#             array = np.array(mask_image, dtype=np.uint8)

#             mask = (array > 0).astype(np.uint8)
            return mask_image
        else:
            print("Failed to connect to the server")
#     def call_rotate(ref_image, test_image):
#             """
#             'call_rotate' is a tool that adjusts the orientation of the reference image using the test image.
#             It aligns the reference image with the test image and returns the rotated image as a PIL.Image object.
    
#             Parameters:
#                 ref_image (PIL.Image): The reference image.
#                 test_image (PIL.Image): The test image used for alignment.
            
#             Returns:
#                 PIL.Image: The rotated reference image.
    
#             Example:
#             --------
#                 >>> rotated = call_rotate(ref_image, test_image)
#             """
#             headers = {'Content-Type': 'application/json'}
#             url = "http://34.59.120.186:8003/rotate"  # Adjust the URL to your deployment
#             payload = {
#                 "ref": convert_image_to_base64(ref_image),
#                 "test": convert_image_to_base64(test_image)
#             }
    
#             tic = time.time()
#             response = requests.post(url, json=payload, headers=headers)
#             toc = time.time()
#             print(f"Rotate service took: {round(toc-tic, 3)} seconds")
    
#             if response.status_code == 200:
#                 result = response.json()
#                 return convert_base64_to_image(result['rotated'])
#             else:
#                 print("Failed to connect to the rotate service")
#                 return None

#     def call_denoise(image):
#             """
#             'call_denoise' is a tool that applies noise removal to the input image.
#             It returns the denoised image as a PIL.Image object.
    
#             Parameters:
#                 image (PIL.Image): The image to be denoised.
    
#             Returns:
#                 PIL.Image: The denoised image.
    
#             Example:
#             --------
#                 >>> denoised = call_denoise(image)
#             """
#             headers = {'Content-Type': 'application/json'}
#             url = "http://34.59.120.186:8003/denoise"  # Adjust the URL to your deployment
#             payload = {
#                 "image": convert_image_to_base64(image)
#             }
    
#             tic = time.time()
#             response = requests.post(url, json=payload, headers=headers)
#             toc = time.time()
#             print(f"Denoise service took: {round(toc-tic, 3)} seconds")
    
#             if response.status_code == 200:
#                 result = response.json()
#                 return convert_base64_to_image(result['denoised'])
#             else:
#                 print("Failed to connect to the denoise service")
#                 return None

#     def call_enhance(image):
#             """
#             'call_enhance' is a tool that applies CLAHE enhancement to a grayscale version of the input image.
#             It returns the enhanced image as a PIL.Image object.
            
#             Parameters:
#                 image (PIL.Image): The image to be enhanced.
    
#             Returns:
#                 PIL.Image: The enhanced image.
    
#             Example:
#             --------
#                 >>> enhanced = call_enhance(image)
#             """
#             headers = {'Content-Type': 'application/json'}
#             url = "http://34.59.120.186:8003/enhance"  # Adjust the URL to your deployment
#             payload = {
#                 "image": convert_image_to_base64(image)
#             }
    
#             tic = time.time()
#             response = requests.post(url, json=payload, headers=headers)
#             toc = time.time()
#             print(f"Enhance service took: {round(toc-tic, 3)} seconds")
    
#             if response.status_code == 200:
#                 result = response.json()
#                 return convert_base64_to_image(result['enhanced'])
#             else:
#                 print("Failed to connect to the enhance service")
#                 return None

#     def call_mask(image1, image2):
#             """
#             'call_mask' is a tool that calculates the residual (difference) between two enhanced images 
#             and generates a binary mask highlighting the differences (defects as 255, background as 0).
#             The final mask is returned as a PIL.Image object.
            
#             Parameters:
#                 image1 (PIL.Image): The first enhanced image.
#                 image2 (PIL.Image): The second enhanced image.
    
#             Returns:
#                 PIL.Image: The generated binary mask.
    
#             Example:
#             --------
#                 >>> mask = call_mask(enhanced_image1, enhanced_image2)
#             """
#             headers = {'Content-Type': 'application/json'}
#             url = "http://34.59.120.186:8003/mask"  # Adjust the URL to your deployment
#             payload = {
#                 "image1": convert_image_to_base64(image1),
#                 "image2": convert_image_to_base64(image2)
#             }
    
#             tic = time.time()
#             response = requests.post(url, json=payload, headers=headers)
#             toc = time.time()
#             print(f"Mask service took: {round(toc-tic, 3)} seconds")
    
#             if response.status_code == 200:
#                 result = response.json()
#                 return convert_base64_to_image(result['mask'])
#             else:
#                 print("Failed to connect to the mask service")
#                 return None

    
    def call_sift_bruteforce_rotate_residual_mask_diff(ref_image, test_image):
        """
        'call_rotate' is a tool that Adjust the orientation of the refence image using the test image. 
        Display where the difference. Supply the end result as a PIL.Image object. 
        It first adjusts the orientation of the reference image using the test image, then performs alignment, 
        denoising, CLAHE enhancement, residual calculation, and mask generation. The resulting mask highlights 
        the regions where differences are present (displaying where the difference is), with potential defects 
        labeled as 255 and all other areas as 0. The final output is supplied as a PIL.Image object.

        Parameters:
            ref_image (PIL.Image): The reference image.
            test_image (PIL.Image): The test image to be compared against the reference image.

        Returns:
            PIL.Image: The generated difference mask as a binary image (defects as 255, background as 0).

        Example:
        --------
        mask = call_rotate(ref_image, test_image)
        """
        headers = {'Content-Type': 'application/json'}
        # 請根據你的部署地址調整 URL
        url = "http://34.59.120.186:8003/sift_bruteforce_rotate_residual_mask_diff"

        # 將圖片轉換為 base64 字串（此處使用 PNG 格式）
        payload = {
            "ref": convert_image_to_base64(ref_image),
            "test": convert_image_to_base64(test_image)
        }

        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"Diff service took: {round(toc-tic, 3)} seconds")

        if response.status_code == 200:
            result = response.json()
            # 將返回的 base64 mask 轉換為 PIL Image 對象
            return convert_base64_to_image(result['mask'])
        else:
            print("Failed to connect to the diff service")
            return None
#     def call_imagen(prompt, number_of_images=1, aspect_ratio="1:1", language="en",
#         safety_filter_level="block_some", person_generation="allow_adult"):
#         """
#         'call_imagen' is a tool that uses Vertex AI's ImageGenerationModel API to generate images based on a         text prompt.
#         It sends a POST request to the image generation service and returns the first generated image as a           PIL.Image object.
    
#         Parameters:
#             prompt (str): A text prompt describing what you want to see.
#             number_of_images (int, optional): Number of images to generate (default: 1).
#             aspect_ratio (str, optional): Desired aspect ratio (default: "1:1").
#             language (str, optional): Language for the prompt (default: "en").
#             safety_filter_level (str, optional): Safety filter level (default: "block_some").
#             person_generation (str, optional): Person generation parameter (default: "allow_adult").
        
#         Returns:
#             PIL.Image: The first generated image.
        
#         Example:
#         --------
#             >>> img = call_imagen("Please draw me 3 cats")
#         """
#         # Adjust the URL to point to your deployed API server (hostname/IP and port)
#         url = "http://34.59.120.186:8004/generate"
    
#         payload = {
#             "prompt": prompt,
#             "number_of_images": number_of_images,
#             "aspect_ratio": aspect_ratio,
#             "language": language,
#             "safety_filter_level": safety_filter_level,
#             "person_generation": person_generation
#         }
#         headers = {'Content-Type': 'application/json'}
    
#         tic = time.time()
#         response = requests.post(url, json=payload, headers=headers)
#         toc = time.time()
#         print(f"call_imagen API call took: {round(toc-tic, 3)} seconds")
    
#         if response.status_code == 200:
#             data = response.json()
#             img_b64 = data.get("generated_image")
#             return convert_base64_to_image(img_b64)
#         else:
#             print("Failed to call the imagen API. Status code:", response.status_code)
#             print(response.text)
#             return None
#     def call_replace_mask_by_generated_image(ref_image, generated_image, boxes):
#         """
#         'call_replace' is a tool that replaces the removed object in the reference image with a generated         dog image.
#         It sends a POST request to the /replace API endpoint with the reference image, the generated dog             image, 
#         and the bounding boxes indicating where the object is located.
#         The API returns the final composited image as a base64-encoded string, which is converted to a             PIL.Image object.
    
#         Parameters:
#             ref_image (PIL.Image): The reference image with the object removed.
#             generated_image (PIL.Image): The generated  image.
#             boxes (List[List[float]]): A list of bounding boxes (each in the format [xmin, ymin, xmax,             ymax]) 
#                                    indicating where to place the dog image.
        
#         Returns:
#             PIL.Image: The final composited image.
    
#         Example:
#         --------
#             >>> final_image = call_replace(ref_image, generated_image, [[50, 30, 150, 130]])
#         """
#         url = "http://34.59.120.186:8003/replace_mask_by_generated_image"  # Adjust URL to your server's address and port
#         payload = {
#             "ref_image": convert_image_to_base64(ref_image),
#             "generated_image": convert_image_to_base64(generated_image),
#             "boxes": boxes
#         }
#         headers = {'Content-Type': 'application/json'}
    
#         tic = time.time()
#         response = requests.post(url, json=payload, headers=headers)
#         toc = time.time()
#         print(f"call_replace API call took: {round(toc-tic, 3)} seconds")
            
#         if response.status_code == 200:
#             data = response.json()
#             final_b64 = data.get("final_image")
#             if final_b64:
#                 return convert_base64_to_image(final_b64)
#             else:
#                 print("No final_image in the response.")
#                 return None
#         else:
#             print("Failed to call the replace API. Status code:", response.status_code)
#             print(response.text)
#             return None
#     def call_remove_mask(image, mask):
#         """
#         'call_remove_mask' is a tool that removes the masked region from an input image by inpainting.
#         It sends a POST request to the /remove_mask API endpoint with the image and its mask as base64-                encoded strings,
#         and returns the inpainted image as a PIL.Image object.

#         Parameters:
#             image (PIL.Image): The input image.
#             mask (PIL.Image): The mask image with regions to remove marked as 255.

#         Returns:
#             PIL.Image: The inpainted image with the masked regions removed.

#         Example:
#         --------
#             >>> result_image = call_remove_mask(input_image, mask_image)
#         """
#         url = "http://34.59.120.186:8003/remove_mask"  # Adjust the URL to your server's address 
    
#         payload = {
#             "image": convert_image_to_base64(image),
#             "mask": convert_image_to_base64(mask)
#         }
#         headers = {'Content-Type': 'application/json'}
    
#         tic = time.time()
#         response = requests.post(url, json=payload, headers=headers)
#         toc = time.time()
#         print(f"call_remove_mask API call took: {round(toc-tic, 3)} seconds")
    
#         if response.status_code == 200:
#             data = response.json()
#             result_b64 = data.get("result_image")
#             if result_b64:
#                 return convert_base64_to_image(result_b64)
#             else:
#                 print("No result_image found in the response.")
#                 return None
#         else:
#             print("Failed to call the remove_mask API. Status code:", response.status_code)
#             print(response.text)
#             return None

#     def call_extract_foreground(generated_image, mask):
#         """
#         'call_extract_foreground' is a tool that extracts the foreground object from a generated image         using a provided mask.
#         It sends a POST request to the /extract_foreground API endpoint and returns the isolated                 foreground (as a PIL.Image object).

#         Parameters:
#         generated_image (PIL.Image): The generated image containing both object and background.
#         mask (PIL.Image): A binary mask image where the object is marked as 255 and background as 0.

#         Returns:
#         PIL.Image: The resulting image with the background removed (foreground isolated with                 transparency).

#         Example:
#         --------
#         >>> fg_image = call_extract_foreground(generated_image, mask)
#         >>> fg_image.show()
#         """
#         url = "http://34.59.120.186:8003/extract_foreground"  # Adjust URL as needed
#         payload = {
#             "generated_image": convert_image_to_base64(generated_image),
#             "mask": convert_image_to_base64(mask)
#         }
#         headers = {'Content-Type': 'application/json'}
    
#         tic = time.time()
#         response = requests.post(url, json=payload, headers=headers)
#         toc = time.time()
#         print(f"call_extract_foreground API call took: {round(toc-tic, 3)} seconds")
    
#         if response.status_code == 200:
#             data = response.json()
#             fg_b64 = data.get("foreground")
#             if fg_b64:
#                 return convert_base64_to_image(fg_b64)
#             else:
#                 print("No foreground image found in the response.")
#                 return None
#         else:
#             print("Failed to call the extract_foreground API. Status code:", 
    def call_diffusion_inpaint(image, mask, prompt=""):
        """'call_diffusion_inpaint' is a tool that performs inpainting using a diffusion model.
        Given an image and a mask, it fills in the masked area based on a text prompt.

        Parameters:
            image (PIL.Image): The input image where certain areas need to be inpainted.
            mask (PIL.Image): The binary mask image where white (255) areas indicate the regions to be inpainted.
            prompt (str, optional): A text description of what should be inpainted in the masked area. Default is an empty string.

        Returns:
            PIL.Image: The inpainted image with the missing regions filled in.
        """
        url = "http://34.59.120.186:8006/diffusion_inpaint"
        payload = {
            'img_base64': convert_image_to_base64(image.convert("RGB")),  # 確保 `RGB`
            'mask_base64': convert_image_to_base64(mask.convert("L")),    # 確保 `L`
            'prompt': prompt
        }

        headers = {'Content-Type': 'application/json'}
        tic = time.time()
        response = requests.post(url, json=payload, headers=headers)
        toc = time.time()
        print(f"Diffusion Inpainting spent: {round(toc-tic, 3)} s")

        if response.status_code == 200:
            output = response.json()
            if 'output_base64' not in output:
                print("Error: 'output_base64' missing in response")
                return None

            inpainted_image = convert_base64_to_image(output['output_base64']).convert("RGB")
            if inpainted_image is None:
                print("Error: Base64 conversion failed")
                return None

            return inpainted_image
        else:
            print(f"Failed to connect to the server, status code: {response.status_code}")
            return None