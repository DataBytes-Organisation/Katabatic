# Import the TableGANAdapter class from the tablegan_adapter module
from .tablegan_adapter import TableGANAdapter
from .tablegan import TableGAN
from .tablegan_utils import preprocess_data, postprocess_data


# Define what should be imported when using "from katabatic.models.tablegan import *"
__all__ = ['TableGANAdapter', 'TableGAN', 'preprocess_data', 'postprocess_data']

# You can add any initialization code for the tablegan module here if needed

# For example, you could set up logging for the tablegan module:
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info("TableGAN module initialized")