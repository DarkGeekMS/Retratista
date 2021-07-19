import io
import sys
import base64
import requests
from PIL import Image


with open(sys.argv[1], "rb") as image_file:
    input_img = base64.b64encode(image_file.read()) 

resp = requests.post(
    "http://localhost:5001/rotate", json={
        "image": input_img,
        "angle" : 30
    }
)

image = base64.b64decode(str(resp.json()['face']))

img = Image.open(io.BytesIO(image))

img.show()
