from io import BytesIO

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow import keras

app = FastAPI()

# CORS middleware configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
MODEL = tf.keras.models.load_model("./savedModels/1")

# Define class names
CLASS_NAMES = ['miner', 'nodisease', 'phoma', 'rust']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

async def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.convert('RGB')  # Ensure RGB format
    image = image.resize((256, 256), Image.LANCZOS)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    print("Shape of preprocessed image:", image.shape)  # Print shape
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Preprocess uploaded image
    image = await preprocess_image(BytesIO(await file.read()))
    # Expand dimensions to match model input shape
    img_batch = np.expand_dims(image, 0)
    # Make predictions
    predictions = MODEL.predict(img_batch)
    # Get predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host='localhost', port=8000)
