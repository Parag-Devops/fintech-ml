import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from sklearn.model_selection import train_test_split
import os.path

def load_data_test() :
    # data = pd.read_csv("..\\mlweb\\input\\paysim1\\PS_20174392719_1491204439457_log.csv")
    data = pd.read_csv("..\\mlweb\\input\\bs140513_032310.csv")
    # data = pd.read_csv("..\\input\\bsNET140513_032310.csv")
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    return train, test

df_train, df_test = load_data_test()
#print(dt['target_names'])
print(df_train.head())
print(df_test.head())

column_descriptions_1 = {
    # 'step': 'ignore', 
    'customer': 'nlp', 
    # 'age': 'nlp', 
    'gender': 'categorical', 
    # 'zipcodeOri': 'nlp', 
    'merchant': 'nlp', 
    # 'zipMerchant': 'nlp', 
    'category': 'nlp', 
    # 'amount': 'nlp',
    'fraud': 'output'
}

column_descriptions_3 = {
                        # 'step', 
                        'type':'ignore', 
                        # 'amount', 
                        'nameOrig':'nlp', 
                        # 'oldbalanceOrg', 
                        # 'newbalanceOrig':'ignore', 
                        'nameDest':'nlp', 
                        # 'oldbalanceDest', 
                        # 'newbalanceDest',	
                        'isFraud':'output', 
                        'isFlaggedFraud':'ignore'
                        }

column_descriptions_2 = {
    'Source': 'nlp', 'Target':'nlp', 'Weight': 'nlp', 'typeTrans': 'nlp', 'fraud': 'output'}

ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions_1)

ml_predictor.train(df_train, model_names='RandomForestClassifier')

# ml_predictor.score(df_test, df_test.fraud)
ml_predictor.score(df_test, df_test.fraud)

ml_predictor.save(file_name="..\\mlweb\\trained_pipeline\\forest\\1.sav")