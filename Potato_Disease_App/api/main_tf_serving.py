from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

## TO connect docker : docker run -t --rm -p 8501:8501 -v C:\Users\Altair\PythonCodes\ML_Learn_Projects\Potato_Disease_App:/Potato_Disease_App tensorflow/serving --rest_api_port=8501 --model_config_file=/Potato_Disease_App/models.config"

app = FastAPI() # Create an instance of fast api


endpoint = "http://localhost:8501/v1/models/CNN_potato_model_v1:predict"


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am here"



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) # Read bytes as a pillow image
    return image
  
    
    
# CReate the brepidct function that uses the CNN model
@app.post("/predict")
async def predict( # using async and await the app will put the preidct function into ascend mode to wait if many users calling the same time
    file: UploadFile = File(...)):
    #COnvert file into Numpy array
    image = read_file_as_image(await file.read())
    #Because model takes batch images we need to expand the iamge dimensions
    img_batch = np.expand_dims(image, 0)
    
    json_data = {
        "instances": img_batch.tolist()
    }
    
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }
    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost' , port = 8000)