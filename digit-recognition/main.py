from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from model import load_model, predict_digit

app = FastAPI(
    title="Digit Recognition API",
    description="An API to recognize handwritten digits from any uploaded image by preprocessing and predicting using a trained model.",
    version="1.0.0",
)

# Load the trained model at startup
model = load_model()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the digit from an uploaded image.
    Accepts any image, converts it to grayscale, resizes it to 28x28 pixels, and normalizes pixel values.
    """
    try:
        # Check file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Only PNG or JPEG images are supported.",
            )

        # Read and process the uploaded image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert(
            "L"
        )  # Convert to grayscale

        # Ensure the image is at least 2D before resizing
        image_array = np.array(image)
        if image_array.ndim < 2:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is invalid or too small.",
            )

        # Resize to 28x28 pixels
        image_resized = image.resize((28, 28))

        # Normalize pixel values to [0, 1]
        image_array = np.array(image_resized, dtype=np.float32) / 255.0

        # Add batch dimension for model input
        image_array = np.expand_dims(
            image_array, axis=(0, -1)
        )  # Shape: (1, 28, 28, 1)

        # Validate input shape
        if image_array.shape != (1, 28, 28, 1):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape: {image_array.shape}. Expected (1, 28, 28, 1).",
            )

        # Make a prediction
        digit, confidence = predict_digit(model, image_array)

        return JSONResponse(
            {"digit": int(digit), "confidence": f"{confidence:.2f}"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}"
        )


@app.get("/")
def root():
    return {
        "message": "Welcome to the Digit Recognition API! Go to /docs for more information."
    }
