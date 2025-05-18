 # Get demo data from GANBLR package
from pandas import read_csv
from katabatic import Katabatic
ganblr_demo_data = read_csv('https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv',dtype=int)
# print(ganblr_demo_data)

X_train, X_test, y_train, y_test = Katabatic.preprocessing(ganblr_demo_data)

print(X_test)

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# # Real Data
# real_data = pd.read_csv('cities_demo.csv')

# # Synthetic Data
# synthetic_data = pd.read_csv('ganblr_output.csv')
# print("Columns Before : ", synthetic_data.columns)
# synthetic_data.pop('Unnamed: 0')
# synthetic_data.columns = ["Temperature", "Longitude", "Latitude","Category"]
# print("Columns After: ", synthetic_data.columns)


# X_real, y_real = real_data[["Temperature","Longitude"]], real_data["Category"]
# X_synthetic, y_synthetic = synthetic_data[["Temperature","Longitude"]], synthetic_data["Category"] #TODO: split x and y

# # Prototype Evaluation Method
# def evaluate(X_real, y_real, X_synthetic, y_synthetic):
#     # TODO: error handling in the case where feature names are missing/do not match
#     # Encode the Real Data
#     # ordinal_enc = OrdinalEncoder()
#     # label_enc  = LabelEncoder()                                            
#     # X_real, y_real = ordinal_enc.transform(X_real), label_enc.transform(y_real)

#     # categories = ["Category"] #['Continental','Subtropical','Tropical']
#     ohe = OneHotEncoder(handle_unknown='ignore')
#     logreg = LogisticRegression()
#     eval_pipeline = Pipeline([('encoder', ohe), ('model', logreg)])
#     eval_pipeline.fit(X_synthetic, y_synthetic)
#     y_pred = eval_pipeline.predict(X_real)

#     return accuracy_score(y_real, y_pred)

# result = evaluate(X_real, y_real, X_synthetic, y_synthetic)
# print("accuracy score: ", result)
        



