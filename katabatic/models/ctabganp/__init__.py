# Import the TableGANAdapter class from the tablegan_adapter module
from .ctabganp_adapter import CTABGANPAdapter
from .ctabganp import CTABGANP
from .ctabganp_utils import preprocess_data, postprocess_data
from ..ctabganp.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics


from .ctabganp import CTABGANP

from ..ctabganp.synthesizer.transformer import ImageTransformer,DataTransformer
from ..ctabganp.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent

from ..ctabganp.pipeline.data_preparation import DataPrep
from ..ctabganp.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

# Define what should be imported when using "from katabatic.models.ctabganp import *"
__all__ = ['CTABGANPAdapter', 'CTABGANP', 'preprocess_data', 'postprocess_data']

# You can add any initialization code for the ctabganp module here if needed

# For example, you could set up logging for the ctabganp module:
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info("CTABGANP module initialized")