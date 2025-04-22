import numpy as np
import base64
import io
from io import BytesIO
import os
import yaml
from skimage.morphology import square, binary_erosion
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import json

def imshow(im):
    plt.imshow(im, cmap = 'gray', interpolation="nearest")
    plt.show()

def imsave(im, path = None):
    im = Image.fromarray(im)
    if im.mode == 'F':
        im = im.convert('RGB')
    if path is None:
        path = './test.png'
    im.save(path)

def imhist(img, path = None, bins=255, range=(0,255)):
    if path is None:
        path = './hist.png'
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(img.ravel(), bins=bins, range=range, fc='k', ec='k')
    fig.savefig(path)
    plt.close(fig)

def replace_edge(image, remove_range, replace_value = 0):
    if remove_range > (np.min(image.shape)-1):
        return image
    for idx in range(remove_range):
        image[idx,:]   = replace_value
        image[-idx-1, :] = replace_value
        image[:, idx]  = replace_value
        image[:, -idx-1] = replace_value
    return image

def get_remove_edge_mask(image, remove_range):
    mask = np.zeros_like(image, dtype = 'bool')
    mask[:,:] = True
    for idx in range(remove_range):
        mask[idx,:]   = False
        mask[-idx-1, :] = False
        mask[:, idx]  = False
        mask[:, -idx-1] = False
    return mask

def get_remove_dx_dy_mask(image, row, column):
    mask = np.zeros_like(image, dtype = 'bool')
    mask[:,:] = True
    
    if row > 0:
        mask[:row,:] = False
    elif row < 0:
        mask[row:,:] = False
        
    if column > 0:
        mask[:,:column] = False
    elif column < 0:
        mask[:,column:] = False
    return mask

def get_border_mask(roi_mask, dx, dy, border_size):
    border_mask = get_remove_edge_mask(roi_mask, border_size) == False
    if border_size > 0:
        dx = int(np.sign(dx)*(np.abs(dx)+border_size+1))
        dy = int(np.sign(dy)*(np.abs(dy)+border_size+1))
        roi_border = get_remove_dx_dy_mask(roi_mask, dx, dy) == False
        border_mask = border_mask | roi_border
    return border_mask

def get_width_by_bbox(bbox):
    (min_row, min_col, max_row, max_col) = bbox
    x_width, y_width = max_col-min_col, max_row-min_row
    return x_width, y_width

def get_width_length_by_mask(regionmask, get_xy=False):
    indices = np.where(regionmask == True)
    y, x = indices
    x_length = max(x)-min(x)+1
    y_length = max(y)-min(y)+1
    if get_xy:
        return x_length, y_length
    width = min(x_length, y_length)
    length = max(x_length, y_length)
    return width, length

def get_scale(image, roi_mask = None):
    if (roi_mask is None) or (not np.any(roi_mask)):
        image = np.array(image, dtype = 'float32')
        if image.dtype == 'bool':
            image = np.uint8(image) * 255
        else:
            image = image - np.min(image)
            if np.sum(image) != 0:
                image = image / np.max(image)*255
        image = image.astype('uint8')
    else:
        image = np.array(image, dtype = 'float32')
        if np.max(image[roi_mask]) == 1:
            image[image == 1] = 255
        else:
            image = image - np.min(image[roi_mask])
            if np.sum(image) != 0:
                image = image / np.max(image[roi_mask])*255
        image = image.astype('uint8')
    return image

def get_transparent_highlight_img(img, filter, rgb):
    for ch, value in enumerate(rgb):
        (img[:,:,ch])[filter] += value
    return img

def get_highlight_img(img, filter, rgb):
    for ch, value in enumerate(rgb):
        (img[:,:,ch])[filter] = value
    return img

def load_yaml(filename):
    """Load configurations in a yaml file and return as a Python object."""
    if not isinstance(filename, str):
        return yaml.safe_load(filename)
    with open(filename) as f:
        return yaml.safe_load(f)

def get_file_b64str(imgpath, if_not_exist=''): # actually it can be any file, i.g. text file
    if os.path.isfile(imgpath):
        with open(imgpath, "rb") as imgfile:
            encoded_string = base64.b64encode(imgfile.read()).decode("utf-8")
    else:
        encoded_string = if_not_exist
    return encoded_string
    
def get_base64(image):
    image = np.array(image, dtype = 'uint8')
    image = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue() 
    image_base64 = base64.b64encode(img_byte_arr)
    image_base64 = image_base64.decode('utf-8')
    return image_base64

def b64_to_bytes(base64str):
    bs = base64.b64decode(base64str)
    bs = BytesIO(bs)
    return bs

def decode_base64_img(image_base64):
    img_bytes = base64.b64decode(image_base64)
    image = np.array(Image.open(io.BytesIO(img_bytes)))
    return image

def decode_base64_pil_img(image_base64):
    img_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(img_bytes))
    return image

def decode_base64(file_base64):
    file_bytes = base64.b64decode(file_base64)
    file = io.BytesIO(file_bytes)
    return file

def convert_base64_to_image(image_data_base64):
    image_data = base64.b64decode(image_data_base64)
    image_bytes = BytesIO(image_data)
    image = Image.open(image_bytes)
    return image

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def make_square(M, val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

def check_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
def print_bar(num, length, bar_width=50, text = ' ', end = False):
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y %H:%M:%S")
    percent = int(num/length*100)
    bar_l = int(percent*bar_width/100)
    bar = '█'* bar_l+ ' '*(bar_width-bar_l)
    if not end:
        print(f'{percent}%|{bar}| {num}/{length} [{current_time}]'+ '\t' + text, end = '\r')
    else:
        print(f'{percent}%|{bar}| {num}/{length} [{current_time}]'+ '\t' + text, end = '\n')
        
def check_bool(str_in):
    if type(str_in) == bool:
        return str_in
    if type(str_in) != str:
        return False
    if str_in.lower() == 'true':
        return True
    else:
        return False
    
def get_bbox(img):
    # y0, x0, y1, x1
    a = np.where(img != 0)
    if np.any(a):
        bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    else:
        bbox = None
    return bbox

def check_config_thr(thr, image):
    # get quantile thr if thr is string startswith q, i.e. q0.05
    if isinstance(thr, str) and thr.startswith('q'):
        quantile_thr = float(thr[1:])
        thr = np.quantile(image, quantile_thr)
    return thr

def dump_json(dictionary, path='./test.json'):
    with open(path, "w") as outfile:
        json.dump(dictionary, outfile)

def parser_np(result):
    for i in result:
        if type(result[i]) is dict:
            result[i] = parser_np(result[i])
        if type(result[i]) is list:
            for k in range(len(result[i])):
                result[i][k] = parser_np({'token': result[i][k]})['token']
        if type(result[i]).__module__ == np.__name__:
            result[i] = result[i].tolist()
        if  type(result[i]).__module__ == Image.__name__:
            result[i] = np.array(result[i]).tolist()
    return result