from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf 

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"



MODEL = tf.keras.models.load_model("/Users/gangminlee/Documents/02.외부활동/03.개인프로젝트/16.Potato/potatoes.h5")
CLASS_NAMES = ['EARLY BLIGHT','Late Blight','Healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive" # check server is alive. 

def read_file_as_image(data)-> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) # PIL image and need to convert it to numpy 
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == '__main__':
    uvicorn.run(app,host = 'localhost',port=8000)

