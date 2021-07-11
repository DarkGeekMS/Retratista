import io
import base64
import requests
from PIL import Image
 

resp = requests.post(
    "http://localhost:5000/refine", json={
        "type"  : "attribute",
        "index" : 7,
        "offset" : -0.5
    }
)

image = base64.b64decode(str(resp.json()['face']))

img = Image.open(io.BytesIO(image))

img.show()
