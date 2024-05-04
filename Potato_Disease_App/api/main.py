from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI() # Create an instance of fast api


MODEL_PATH = "Potato_DIsease_App/models/CNN_potato_model_v1.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
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
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {'class' : predicted_class,
            'confidence' : float(confidence)}
    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost' , port = 8000)