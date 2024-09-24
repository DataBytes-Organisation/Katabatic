from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import pandas as pd
import logging
import numpy as np
from katabatic.katabatic import Katabatic
from katabatic.models.ganblrpp_DGEK.ganblrpp_adapter import GanblrppAdapter

# Initialize FastAPI app
app = FastAPI(
    title="GANBLR++ Model API",
    description="An API to train and generate synthetic data using the GANBLR++ model. "
    "Provides endpoints to train the model on a dataset and generate synthetic data.",
    version="1.0.0",
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the GANBLR++ model instance (the model instance will persist across requests)
ganblrpp_model = None

def initialize_model(numerical_columns):
    global ganblrpp_model
    ganblrpp_model = GanblrppAdapter(numerical_columns=numerical_columns)

# Request body for generating synthetic data
class GenerateRequest(BaseModel):
    num_samples: int  # Number of samples to generate


@app.post("/train", summary="Train the GANBLR++ model", tags=["Model Operations"])
async def train_model(
    file: UploadFile = File(...),
    epochs: int = Form(...),
    batch_size: int = Form(64)
):
    """
    Train the GANBLR++ model on the provided dataset.

    - **file**: CSV dataset file.
    - **epochs**: Number of epochs to run the training.
    - **batch_size**: (Optional) Batch size for training.

    This will train the model and persist it in memory for future synthetic data generation.
    """
    try:
        # Load dataset
        logging.info(f"Loading dataset from uploaded file...")
        data = pd.read_csv(file.file)

        # Identify numerical columns
        def is_numerical(dtype):
            return dtype.kind in 'iuf'

        column_is_numerical = data.dtypes.apply(is_numerical).values
        numerical = np.argwhere(column_is_numerical).ravel()

        if len(numerical) == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain at least one numerical feature.",
            )

        # Initialize model with numerical columns
        initialize_model(numerical)

        # Split the dataset into features (X) and labels (y)
        if len(data.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain at least one feature and one label.",
            )

        X, y = data.iloc[:, :-1], data.iloc[:, -1]

        # Train the model
        logging.info(f"Training the GANBLR++ model for {epochs} epochs with batch size {batch_size}...")
        ganblrpp_model.load_model()  # Load the model if necessary
        ganblrpp_model.fit(X, y, epochs=epochs, batch_size=batch_size)

        logging.info("Model trained successfully.")
        return {"message": "Model trained successfully", "epochs": epochs}
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during training: {str(e)}"
        )


@app.post("/generate", summary="Generate synthetic data", tags=["Model Operations"])
async def generate_synthetic_data(request: GenerateRequest):
    """
    Generate synthetic data from the trained GANBLR++ model.

    - **num_samples**: Number of synthetic samples to generate.

    Returns a JSON response with the generated synthetic data.
    """
    try:
        if ganblrpp_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not trained. Please train the model first."
            )

        logging.info(f"Generating {request.num_samples} synthetic data samples...")
        synthetic_data = ganblrpp_model.generate(size=request.num_samples)
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
    return {"message": "GANBLR++ Model API is running!"}

