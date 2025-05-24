# MEG Project

This project implements a Generative Model based on Masked Generators and Discriminators. It can be trained on custom datasets like the Adult dataset or any other dataset for generative modeling.

## Requirements
- Python 3.9
- Install dependencies via `pip install -r requirements.txt`

## Project Structure
- **MEG_Adapter.py**: Main script for training and adapting the models.
- **MaskedGenerator.py**: The generator model definition.
- **Discriminator.py**: The discriminator model definition.
- **utils.py**: Utility functions for saving/loading models.
- **data/**: Folder to store datasets.
- **experiments/**: Folder for experiments and logs.

## How to Run
1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run the main script to start training:
    ```bash
    python MEG_Adapter.py
    ```

## Datasets
The project uses the **Adult** dataset located in the `data/` folder. You can replace it with your own dataset for training.

## Experiment Logs
Logs for different runs will be saved in the `experiments/logs/` folder.
