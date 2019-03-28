import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import os.path

jsonpath = "..\\mlweb\\static\\json\\train_1.json"

tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

# df_train, df_test = get_boston_dataset()

column_descriptions2 = {
'Carhdolder Name' : 'nlp',
'Transaction Date' : 'date',
# 'Transaction Amount',
'Merchant Category Code Description' : 'nlp',
'Merchant Name' : 'nlp',
'Merchant State/Province' : 'nlp',
'Department' : 'nlp',
'Department Contact' : 'nlp',
'Phone #' : 'nlp'
 }

column_descriptions = {
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

column_descriptions_2 = {
    'Source': 'nlp', 'Target':'nlp', 'Weight': 'nlp', 'typeTrans': 'nlp', 'fraud': 'output'}

class deepl(object):
    def __init__(self):
        self.train_df = pd.read_csv("..\\mlweb\\input\\bs140513_032310.csv")
        # self.data = pd.read_csv("..\\input\\bsNET140513_032310.csv")
        self.ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
        self.model = Sequential()

    def handle_non_numerical_data(self):
        columns = self.train_df.columns.values
        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]

            if self.train_df[column].dtype != np.int64 and self.train_df[column].dtype != np.float64:
                column_contents = self.train_df[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1

                self.train_df[column] = list(map(convert_to_int, self.train_df[column]))

    def getX_Y(self):
        return train_test_split(self.train_df, test_size=0.2, shuffle=True)

    def get_train_X_Y(self):
        train_X = self.train_df.drop(columns = ['fraud'])
        train_Y = self.train_df[['fraud']]
        print(train_X.head())
        print(train_Y.head())
        return train_X, train_Y

    def create_dl_model(self):
        train_X, train_Y = self.get_train_X_Y()
        #get number of columns in training data
        n_cols = train_X.shape[1]
        #add model layers
        self.model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def get_dl_model(self):
        return self.model

    def train_dl_model(self):
        train_X, train_Y = self.get_train_X_Y()
        #set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=3)
        #train model
        self.model.fit(train_X, train_Y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

    def sav_dl_model(self):
        self.model.save_weights('..\\mlweb\\trained_pipeline\\deepl\\deep_learning.h5')

    def get_features(self):
        return list(self.train_df)

    def send_tojson(self, list):
        features = {}
        list = self.get_features()
        for x in range(len(list)):
            features["feature" + str(x)] = list[x]
        with open(jsonpath, 'w') as outfile:  
            json.dump(features, outfile)
                
    def learn_model(self):
        df_train, df_test = self.getX_Y()
        # self.ml_predictor.train(df_train, model_names='DeepLearningRegressor')
        self.ml_predictor.train(df_train, feature_learning = True, fl_data = df_test, model_names ='DeepLearningRegressor')
        self.ml_predictor.score(df_test, df_test.fraud)

    def sav_model(self):
        self.ml_predictor.save()


# dmodel = deepl()
# dmodel.send_tojson(dmodel.get_features())
# # dmodel.getX_Y()
# dmodel.learn_model()
# dmodel.sav_model()

#2
# dmodel = deepl()
# dmodel.handle_non_numerical_data()
# dmodel.create_dl_model()
# dmodel.train_dl_model()
# dmodel.sav_dl_model()

# print(dmodel.get_features())

# SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
# pipline_path = os.path.join(SITE_ROOT, "trained_pipeline/deepl", "deepLearning.h5")
# print(pipline_path)