import io
import base64
import requests
from PIL import Image
 

resp = requests.post(
    "http://localhost:5000/vgenerate", json={
        "values": [-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0,
                3.3894314765930176, -100.0, -100.0, -100.0, -100.0, -100.0,
                -100.0, 4.0, -0.9020038604736329, -100.0, -100.0, -100.0,
                -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0,
                -100.0, -100.0, -100.0, -100.0, 3.627786636352539, -100.0]
    }
)

print(resp.json()['values'])

image = base64.b64decode(str(resp.json()['face']))

img = Image.open(io.BytesIO(image))

img.show()
