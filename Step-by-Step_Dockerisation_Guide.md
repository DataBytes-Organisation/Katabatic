# GANBLR Model API

A **FastAPI** service to train and generate synthetic data using the **GANBLR** model. This service allows you to train the GANBLR model on your dataset and generate synthetic data from the trained model.

## Features

- **Train the Model**: Upload a dataset and specify the number of epochs for training.
- **Generate Synthetic Data**: Generate synthetic data after the model is trained.
- **REST API Interface**: Access endpoints to train and generate data using FastAPI.
- **Modular Setup**: Run the service using a virtual environment or through a Docker container.

---

## Quickstart Guide

### Prerequisites

Before you get started, ensure that you have the following tools installed:

- **Python 3.9.19**
- **Conda** (for environment management)
- **Docker** (optional, for Dockerized deployment)

---

## Running the API in a Local Environment

### Step 1: Setup the Environment

We recommend using **Conda** to manage the environment and ensure all dependencies are installed in the correct order.

#### Bash Script

You can use the provided `serve_ganblr.sh` script to create a conda environment, install dependencies, and serve the FastAPI application:

```bash
#!/bin/bash

# Step 1: Setup conda environment
conda create --name ganblr_env python=3.9.19 -y
conda activate ganblr_env

# Step 2: Install libraries in the correct order
echo "Installing required libraries..."
pip install pyitlib
pip install tensorflow
pip install pgmpy
pip install sdv
pip install scikit-learn==1.0
pip install fastapi uvicorn

# Step 3: Serve the model API
uvicorn ganblr_api:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Run the Bash Script

Save the bash script as serve_ganblr.sh, then run the following command:

```bash
chmod +x serve_ganblr.sh
./serve_ganblr.sh
```

This will set up the environment, install dependencies, and start the FastAPI server. You can access the API at http://localhost:8000.

### Step 3: API Endpoints

    •	Training the Model

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'Content-Type: application/json' \
  -d '{
  "file_path": "/path/to/your/dataset.csv",
  "epochs": 20
}'
```

    •	Input:
    •	file_path: The path to your dataset in CSV format.
    •	epochs: The number of training epochs.
    •	Output:
    •	A success message indicating the model was trained.

    •	Generating Synthetic Data

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{
  "num_samples": 100
}'
```

    •	Input:
    •	num_samples: The number of synthetic samples to generate.
    •	Output:
    •	JSON array containing synthetic data.

## Running the API with Docker

If you’d rather run the API in a Docker container, follow these steps.

### Step 1: Create a Dockerfile

The following Dockerfile is already provided in the repository and is set up to install dependencies and serve the FastAPI API.

#### Step 1: Use an official Miniconda image as a base image

```bash
FROM continuumio/miniconda3:latest
```

#### Step 2: Set the working directory in the container

```bash
WORKDIR /app
```

#### Step 3: Copy your application code (GANBLR model API code)

```bash
COPY . /app
```

#### Step 4: Create a new Conda environment

```bash
RUN conda create --name ganblr_env python=3.9.19 -y && \
    echo "source activate ganblr_env" > ~/.bashrc
```

#### Step 5: Activate the environment and install necessary libraries in the precise order

```bash
RUN /bin/bash -c "source activate ganblr_env && \
    pip install pyitlib && \
    pip install tensorflow && \
    pip install pgmpy && \
    pip install sdv && \
    pip install scikit-learn==1.0 && \
    pip install fastapi uvicorn"
```

#### Step 6: Expose the port that the FastAPI app will run on

```bash
EXPOSE 8000
```

#### Step 7: Run the FastAPI server

```bash
CMD ["/bin/bash", "-c", "source activate ganblr_env && uvicorn ganblr_api:app --host 0.0.0.0 --port 8000"]
```

### Step 2: Build the Docker Image

From the directory where your Dockerfile and code are located, build the Docker image:

```bash
docker build -t ganblr-fastapi .
```

### Step 3: Run the Docker Container

Once the image is built, run the Docker container with the following command:

```bash
docker run -p 8000:8000 -v /path/to/your/local/directory:/app/data ganblr-fastapi
```

This will:

    •	Expose port 8000: The FastAPI app will be available at http://localhost:8000.
    •	Mount the local directory: It mounts the local directory /path/to/your/local/directory into the container’s /app/data directory.

### Step 4: API Endpoints (Same as Local)

You can now interact with the API at http://localhost:8000.

    •	Training the Model (with dataset file inside Docker):

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'Content-Type: application/json' \
  -d '{
  "file_path": "/app/data/katabatic/nursery/nursery.data",
  "epochs": 20
}'
```

    •	Generating Synthetic Data:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{
  "num_samples": 100
}'
```

## API Documentation

### POST /train

Train the GANBLR model on a dataset provided by the user.

    •	Request Body:
    •	file_path (string): Path to the CSV dataset file.
    •	epochs (integer): Number of epochs for training.
    •	Response:
    •	200: Successful training message.
    •	500: Error during training (e.g., file not found or other issue).

### POST /generate

Generate synthetic data from the trained GANBLR model.

    •	Request Body:
    •	num_samples (integer): Number of synthetic samples to generate.
    •	Response:
    •	200: JSON array with generated synthetic data.
    •	500: Error during data generation.

### GET /

Check the status of the API.

    •	Response:
    •	200: Status message indicating that the API is running.

## Contributing

Contributions are welcome! Please follow these steps:

    1.	Fork the repository.
    2.	Create a new branch (git checkout -b feature-branch).
    3.	Make your changes and ensure tests pass.
    4.	Submit a pull request for review.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or issues, please feel free to open an issue in the repository or reach out to the maintainers.
