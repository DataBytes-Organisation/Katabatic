#install ganblr 
!pip install ganblr

#data loading phase
from ganblr.utils import get_demo_data
df = get_demo_data('adult-raw')
df.head()

#Traning phase 
from sklearn.model_selection import train_test_split
x, y = df.values[:,:-1], df.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

#GANBLR's unique specialicity that also able to includes mumerrical data
import numpy as np
def is_numerical(dtype):
    '''
    if the type is one of ['signed-integer', 'unsigned-integer', 'floating point'], we reconginze it as a numerical one.
    
    Reference: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
    '''
    return dtype.kind in 'iuf'

column_is_numerical = df.dtypes.apply(is_numerical).values
numerical_columns = np.argwhere(column_is_numerical).ravel()
numerical_columns

#Synthesis of the tabular data
from ganblr import GANBLRPP
ganblrpp = GANBLRPP(numerical_columns)
ganblrpp.fit(X_train, y_train, epochs=10)

#output size thus optional 
size = 1000
syn_data = ganblrpp.sample(size)

#view output
import pandas as pd
pd.DataFrame(syn_data, columns=df.columns).head(10)

#Evaluation tools, perofrmance metrices and etc.

acc_score_lr  = ganblrpp.evaluate(X_test, y_test, model='lr')
acc_score_mlp = ganblrpp.evaluate(X_test, y_test, model='mlp')
acc_score_rf  = ganblrpp.evaluate(X_test, y_test, model='rf')

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score

catgorical_columns = list(set(range(X_train.shape[1])) - set(numerical_columns))  

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_ohe = ohe.fit_transform(X_train[:,catgorical_columns])
X_test_ohe  = ohe.transform(X_test[:,catgorical_columns])
X_train_num = X_train[:,numerical_columns]
X_test_num  = X_test[:,numerical_columns]

scaler = StandardScaler()
X_train_concat = scaler.fit_transform(np.hstack([X_train_num, X_train_ohe]))
X_test_concat  = scaler.transform(np.hstack([X_test_num, X_test_ohe]))

lbe = LabelEncoder()
y_train_lbe = lbe.fit_transform(y_train)
y_test_lbe = lbe.transform(y_test)

trtr_score_lr = LogisticRegression().fit(X_train_concat, y_train).score(X_test_concat, y_test)
trtr_score_rf = RandomForestClassifier().fit(X_train_concat, y_train).score(X_test_concat, y_test)
trtr_score_mlp = MLPClassifier().fit(X_train_concat, y_train).score(X_test_concat, y_test)

#output performance results
df_evaluate = pd.DataFrame([
    ['TSTR', acc_score_lr, acc_score_rf, acc_score_mlp],
    ['TRTR', trtr_score_lr,trtr_score_rf,trtr_score_mlp]
], columns=['Evaluated Item', 'LR', 'RF', 'MLP'])
df_evaluate




