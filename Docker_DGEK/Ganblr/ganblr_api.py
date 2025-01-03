from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import logging
from katabatic.katabatic import Katabatic
import os

# Initialize FastAPI app
app = FastAPI(
    title="GANBLR Model API",
    description="An API to train and generate synthetic data using the GANBLR model. "
    "Provides endpoints to train the model on a dataset and generate synthetic data.",
    version="1.0.0",
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the GANBLR model instance (the model instance will persist across requests)
ganblr_model = Katabatic.run_model("ganblr")


# Request body for training
class TrainRequest(BaseModel):
    file_path: str  # Path to the dataset file (CSV)
    epochs: int  # Number of epochs for training


# Request body for generating synthetic data
class GenerateRequest(BaseModel):
    num_samples: int  # Number of samples to generate


@app.post("/train", summary="Train the GANBLR model", tags=["Model Operations"])
async def train_model(request: TrainRequest):
    """
    Train the GANBLR model on the provided dataset.

    - **file_path**: Path to the CSV dataset file.
    - **epochs**: Number of epochs to run the training.

    This will train the model and persist it in memory for future synthetic data generation.
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        logging.info(f"Loading dataset from {request.file_path}...")
        data = pd.read_csv(request.file_path)

        # Assuming the last column is the label, split the dataset into features (X) and labels (y)
        if len(data.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain at least one feature and one label.",
            )

        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # Train the model
        logging.info(f"Training the GANBLR model for {request.epochs} epochs...")
        ganblr_model.load_model()  # Load the model if necessary
        ganblr_model.fit(X, y, epochs=request.epochs)

        logging.info("Model trained successfully.")
        return {"message": "Model trained successfully", "epochs": request.epochs}
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during training: {str(e)}"
        )


@app.post("/generate", summary="Generate synthetic data", tags=["Model Operations"])
async def generate_synthetic_data(request: GenerateRequest):
    """
    Generate synthetic data from the trained GANBLR model.

    - **num_samples**: Number of synthetic samples to generate.

    Returns a JSON response with the generated synthetic data.
    """
    try:
        logging.info(f"Generating {request.num_samples} synthetic data samples...")
        synthetic_data = ganblr_model.generate(size=request.num_samples)
        synthetic_df = pd.DataFrame(synthetic_data)

        logging.info("Synthetic data generation successful.")
        return {"synthetic_data": synthetic_df.to_dict(orient="records")}
    except Exception as e:
        logging.error(f"Error during synthetic data generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during data generation: {str(e)}",
        )


@app.get("/", summary="Root Endpoint", tags=["General"])
async def root():
    """
    Root endpoint for checking the status of the API.

    Returns a simple message indicating that the API is up and running.
    """
    return {"message": "GANBLR Model API is running!"}
