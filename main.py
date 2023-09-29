import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

app = FastAPI()

# Enable CORS for all origins (you can configure this to be more restrictive)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["*"] to allow all origins, or specify specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved TensorFlow model
model = tf.keras.models.load_model('./model')
model.summary()  # Print model summary for verification

class InputData(BaseModel):
    data: list

class OutputData(BaseModel):
    prediction: list

# Define mapping dictionary for nature-related activities
nature_activity_mapping = {
    0: "Nature Sketching",
    1: "Leave No Trace Camping",
    2: "Photography",
    3: "Stargazing",
    4: "Native Plant Gardening",
    5: "Environmental Advocacy",
    6: "Staying Eco-Friendly Indoors",
    7: "Wind-Powered Activities",
    8: "Citizen Science",
    9: "Fishing",
    10: "Fog Photography",
    11: "Environmental Education",
    12: "Fog-Responsible Driving",
    13: "Birdwatching",
    14: "Nature Walks",
    15: "Bicycling",
    16: "Cloud Gazing",
    17: "Outdoor Gardening",
    18: "Clean-Up Initiatives",
    19: "Outdoor Gardening",
    20: "Zero-Waste Challenge",
    21: "Community Gardens",
    22: "Nature Photography",
    23: "Outdoor Reading",
    24: "Window Watching",
    25: "Plan a Rainy-Day Garden Party",
    26: "Rainy Photography",
    27: "Rain Gauge Monitoring",
    28: "Water Conservation",
    29: "Environmental Cleanup",
    30: "Energy-Efficient Appliances",
    31: "Building a Snow Fort or Igloo",
    32: "Ice Skating",
    33: "Snowshoeing",
    34: "Winter Birdwatching",
    35: "Cross-Country Skiing",
    36: "Sustainable Shopping",
    37: "RainWater Harvesting",
    38: "Solar Energy Exploration",
    39: "Wildflower and Plant Identification",
}

@app.post("/predict/", response_model=OutputData)
async def predict(input_data: InputData):
    input_array = tf.convert_to_tensor(input_data.data, dtype=tf.float32)
    input_array = tf.expand_dims(input_array, axis=-1)

    # Ensure the input shape matches [1, 5, 1, 1]
    if input_array.shape != (1, 5, 1, 1):
        return {"prediction": []}  # Return an empty list as a valid response

    # Perform inference
    prediction = model.predict(input_array)
    prediction = np.argsort(prediction, axis=1)[:, -4:]
    prediction_list = prediction.tolist()

    # Map numerical predictions to nature-related activity names
    nature_activities = [nature_activity_mapping[label] for label in prediction_list[0]]

    return {
        "prediction": nature_activities,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
