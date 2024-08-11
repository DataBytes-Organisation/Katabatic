from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
from .ganblr import GANBLR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GanblrAdapter(KatabaticModelSPI):
    def __init__(self, type="discrete"):
        self.type = type
        self.model = None
        self.training_sample_size = 0

    def load_model(self):
        logger.info("Initializing GANBLR Model")
        self.model = GANBLR()
        return self.model

    def load_data(self, data_pathname):
        logger.info(f"Loading data from {data_pathname}")
        try:
            data = pd.read_csv(data_pathname)
            logger.info("Data loaded successfully.")
            return data
        except FileNotFoundError:
            logger.error(f"File '{data_pathname}' not found.")
        except pd.errors.EmptyDataError:
            logger.error("The file is empty.")
        except pd.errors.ParserError:
            logger.error("Parsing error while reading the file.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        return None

    def fit(self, X_train, y_train, k=0, epochs=10, batch_size=64):
        try:
            logger.info("Training GANBLR model")
            self.model.fit(
                X_train, y_train, k=k, batch_size=batch_size, epochs=epochs, verbose=0
            )
            self.training_sample_size = len(X_train)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")

    def generate(self, size=None):
        try:
            logger.info("Generating data using GANBLR model")
            if size is None:
                size = self.training_sample_size
            generated_data = self.model.sample(size, verbose=0)
            logger.info("Data generation completed")
            return generated_data
        except Exception as e:
            logger.error(f"An error occurred during data generation: {e}")
            return None
