from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import numpy as np
#from eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
import numpy as np
import pandas as pd
import glob
from .ctabganp import CTABGANP
from .ctabganp_utils import preprocess_data, postprocess_data

class CTABGANPAdapter(KatabaticModelSPI):

    def __init__(self, type, raw_csv_path = "Real_Datasets/Adult.csv"):
        self.real_path = raw_csv_path
        self.type = type  # Should be either 'discrete' or 'continuous'
        self.constraints = None 
        self.batch_size = None
        self.epochs = None

        #self.raw_csv_path = raw_csv_path
        self.num_exp = 5
        self.dataset = "Adult"
        self.fake_file_root = "Fake_Datasets"

    def load_model(self,
                 test_ratio = 0.20,
                 headers = None,
                 sep = None,
                 categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": 'income'},
                 preprocessing_raw_df = None):
        print("---Initialise CTABGANP Model")

        self.model =  CTABGANP(raw_csv_path = self.real_path,
                        headers = headers,
                        sep=sep,
                        test_ratio = test_ratio,
                        categorical_columns = categorical_columns, 
                        log_columns = log_columns,
                        mixed_columns = mixed_columns,
                        general_columns = general_columns,
                        non_categorical_columns = non_categorical_columns,
                        integer_columns = integer_columns,
                        problem_type= problem_type,
                        preprocessing_raw_df=preprocessing_raw_df)

        self.raw_df = self.model.raw_df
        self.x_train = self.model.x_train
        self.y_train = self.model.y_train
        self.x_test = self.model.x_test
        self.y_test = self.model.y_test

    def load_data(self):
        pass

    def fit(self,epochs=100):
        self.model.fit(epochs)

    def generate(self, size=None):
        syn = self.model.generate_samples(size)
        return syn
        '''
        adult_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
        stat_res_avg = []
        for fake_path in self.fake_paths:
            stat_res = stat_sim(self.real_path,fake_path,adult_categorical)
            stat_res_avg.append(stat_res)

        stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
        stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
        print("stat_results:")
        print(stat_results)

        priv_res_avg = []
        for fake_path in self.fake_paths:
            priv_res = privacy_metrics(self.real_path,fake_path)
            priv_res_avg.append(priv_res)
            
        privacy_columns = ["DCR between Real and Fake (5th perc)","DCR within Real(5th perc)","DCR within Fake (5th perc)","NNDR between Real and Fake (5th perc)","NNDR within Real (5th perc)","NNDR within Fake (5th perc)"]
        privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1,6),columns=privacy_columns)
        print("privacy_results:")
        print(privacy_results)
        '''

'''
class CTABGANPAdapter(KatabaticModelSPI):

    def __init__(self):
        #self.raw_csv_path = raw_csv_path
        self.num_exp = 5
        self.dataset = "Adult"
        self.real_path = "Real_Datasets/Adult.csv"
        self.fake_file_root = "Fake_Datasets"

    def load_model(self):
        print("---Initialise CTABGANP Model")

        self.synthesizer =  CTABGANP(raw_csv_path = self.real_path,
                        test_ratio = 0.20,
                        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                        log_columns = [],
                        mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                        general_columns = ["age"],
                        non_categorical_columns = [],
                        integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                        problem_type= {"Classification": 'income'}) 

    def load_data(self, data_pathname):
        print("Loading Data...")
        try:
            data = pd.read_csv(data_pathname)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def fit(self):
        self.synthesizer.fit()
        syn = self.synthesizer.generate_samples()
        syn.to_csv(self.fake_file_root+"/"+self.dataset+"/"+ self.dataset+"_fake_{exp}.csv".format(exp=i), index= False)

        for i in range(self.num_exp):
            self.synthesizer.fit()
            syn = self.synthesizer.generate_samples()
            syn.to_csv(self.fake_file_root+"/"+self.dataset+"/"+ self.dataset+"_fake_{exp}.csv".format(exp=i), index= False)

    def train(self):
        fake_paths = glob.glob(self.fake_file_root+"/"+ self.dataset+"/"+"*")
        model_dict =  {"Classification":["lr","dt","rf","mlp","svm"]}
        result_mat = get_utility_metrics(self.real_path,fake_paths,"MinMax",model_dict, test_ratio = 0.20)

        result_df  = pd.DataFrame(result_mat,columns=["Acc","AUC","F1_Score"])
        result_df.index = list(model_dict.values())[0]
        print("result_df:")
        print(result_df)

    def evaluate(self):
        adult_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
        stat_res_avg = []
        for fake_path in self.fake_paths:
            stat_res = stat_sim(self.real_path,fake_path,adult_categorical)
            stat_res_avg.append(stat_res)

        stat_columns = ["Average WD (Continuous Columns","Average JSD (Categorical Columns)","Correlation Distance"]
        stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1,3),columns=stat_columns)
        print("stat_results:")
        print(stat_results)

        priv_res_avg = []
        for fake_path in self.fake_paths:
            priv_res = privacy_metrics(self.real_path,fake_path)
            priv_res_avg.append(priv_res)
            
        privacy_columns = ["DCR between Real and Fake (5th perc)","DCR within Real(5th perc)","DCR within Fake (5th perc)","NNDR between Real and Fake (5th perc)","NNDR within Real (5th perc)","NNDR within Fake (5th perc)"]
        privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1,6),columns=privacy_columns)
        print("privacy_results:")
        print(privacy_results)
'''