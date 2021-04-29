import io
from base64 import encodebytes
from PIL import Image


def get_response_image(img):
    # encode image for response
    # convert image array to PIL image
    pil_img = Image.fromarray(img)
    # initialize byte array
    byte_arr = io.BytesIO()
    # convert PIL image to byte array
    pil_img.save(byte_arr, format='PNG')
    # encode as base64
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    # return encoded image
    return encoded_img
