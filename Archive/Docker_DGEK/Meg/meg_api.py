from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import pandas as pd
import logging
import numpy as np
from katabatic.models.meg_DGEK.meg_adapter import MegAdapter
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI(
    title="MEG Model API",
    description="An API to train and generate synthetic data using the MEG model. "
    "Provides endpoints to train the model on a dataset and generate synthetic data.",
    version="1.0.0",
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the MEG model instance (the model instance will persist across requests)
meg_model = None
original_labels = None

def initialize_model():
    global meg_model
    meg_model = MegAdapter()

def store_labels(df):
    global original_labels
    original_labels = df.iloc[:, 0].unique()

# Request body for generating synthetic data
class GenerateRequest(BaseModel):
    num_samples: int  # Number of samples to generate

@app.post("/train", summary="Train the MEG model", tags=["Model Operations"])
async def train_model(
    file: UploadFile = File(...),
    epochs: int = Form(...),
):
    """
    Train the MEG model on the provided dataset.

    - **file**: CSV dataset file.
    - **epochs**: Number of epochs to run the training.

    This will train the model and persist it in memory for future synthetic data generation.
    """
    global meg_model, original_labels

    try:
        # Load dataset from uploaded file
        logging.info("Loading dataset from uploaded file...")
        df = pd.read_csv(file.file)

        # Store original labels
        store_labels(df)

        # Split the dataset into features (X) and labels (y)
        if len(df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain at least one feature and one label.",
            )

        X, y = df.values[:, :-1], df.values[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        # Initialize and train model
        initialize_model()
        logging.info(f"Training the MEG model for {epochs} epochs...")
        meg_model.load_model()  # Load the model if necessary
        meg_model.fit(X_train, y_train, epochs=epochs)

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
    Generate synthetic data from the trained MEG model.

    - **num_samples**: Number of synthetic samples to generate.

    Returns a JSON response with the generated synthetic data.
    """
    try:
        if meg_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not trained. Please train the model first."
            )

        if original_labels is None:
            raise HTTPException(
                status_code=500,
                detail="Original labels not available. Please train the model first."
            )

        logging.info(f"Generating {request.num_samples} synthetic data samples...")
        synthetic_data = meg_model.generate(size=request.num_samples)
        synthetic_df = pd.DataFrame(synthetic_data)

        # Revert the first column to its original label
        synthetic_df.iloc[:, 0] = pd.Series(synthetic_df.iloc[:, 0]).map(lambda x: original_labels[int(x) % len(original_labels)])

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
    return {"message": "MEG Model API is running!"}



