import io
import base64
import requests
from PIL import Image
 

resp = requests.post(
    "http://localhost:5000/tgenerate", json={"text":"young bald man with beard and glasses"}
)

print(resp.json()['values'])

image = base64.b64decode(str(resp.json()['face']))

img = Image.open(io.BytesIO(image))

img.show()
