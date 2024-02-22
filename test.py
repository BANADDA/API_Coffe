import requests
from PIL import Image


def preprocess_image(image_file_path):
    with open(image_file_path, "rb") as file:
        image = Image.open(file)
        image = image.resize((256, 256))
    return image

url = "https://api-coffe-tu74.onrender.com/predict"

image_file_path = "miner.png"

image = preprocess_image(image_file_path)
image_bytes = image.tobytes()

files = {"file": image_bytes}

response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.text)
