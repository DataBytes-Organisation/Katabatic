Katabatic
=========================================
Katabatic is an open-source tabular data generation framework designed for data generative models such as GANBLR, TableGAN, MedGan etc...

#### Code Description

Katabatic is a framework designed to make generating and evaluating synthetic data much easier. Katabatic has been build with the understanding that different domains have different requirements of synthetic data, and therefore provides a range of evaluation methods.

#### Installation

1. Install Dependencies

2. Download/clone the Katabatic code

#### Usage

The first step is to import the katabatic library:
	
 	import katabatic

Next, import the desired model or models:

 	from katabatic import GANBLR


#### Relevant Publications

GANBLR is a data generative model introduced in the following [paper](https://www.researchgate.net/publication/356159733_GANBLR_A_Tabular_Data_Generation_Model):

	Zhang, Yishuo & Zaidi, Nayyar & Zhou, Jiahui & li, Gang. (2021). 
 	GANBLR: A Tabular Data Generation Model. 10.1109/ICDM51629.2021.00103. 


  # TO DO:
  - Improve the evaluation methods so they are easier to use.
  - Debug evaluate_data()
  - Move individual models into docker containers.
  - Finalise documentation page "Getting Started with Katabatic"
  - Finalise documentation page "Installation Guide"
  - Cleanup folder structure in preparation to add Katabatic to PyPi.
  - Move Aiko implementation from Prototype to Katabatic

  


