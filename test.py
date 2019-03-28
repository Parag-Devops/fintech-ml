import pandas as pd
from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model
from auto_ml.utils_scoring import advanced_scoring_classifiers
import dill
import keras
import numpy as np # linear algebra
from keras.models import load_model, Sequential
from keras.layers import Dense
import os.path

strpath = "..\\mlweb\\test_files\\deepl_test.csv"

class Test(object):
    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)
        self.trained_model = load_ml_model(file_name="..\\mlweb\\trained_pipeline\\forest\\1.sav")

    def test_forest(self, df_test):
        predictions = self.trained_model.predict(df_test)
        return predictions

    def getScore(self, df_test):
        pred_proba = self.trained_model.predict_proba(df_test)
        # pred_proba = [proba[1] for proba in pred_proba]
        print("Random Forest Probability -->", pred_proba)
        actuals = [0, 1]
        brier_score = advanced_scoring_classifiers(pred_proba, actuals)
        print("*********** Accuracy of Randon Forest Model ***********************")
        print("*** The best possible Brier score is 0, for total accuracy.")
        print("*** The lowest possible score is 1, which mean the forecast was wholly inaccurate.")
        print("Smaller scores (closer to zero) indicate better forecasts. Scores in the middle (e.g. 0.44, 0.69) can be hard to interpret as good or bad")
        print("Brier Score for this test is -->", brier_score)
        print("*********** Accuracy of Randon Forest Model ***********************")
        return brier_score     


# print(os.path.exists("..\\mlweb\\trained_pipeline\\forest\\1.sav"))
# print("Current Dir : ", os.getcwd())

def insert_deep_learning_model(pipeline_step, file_name):
    # This is where we saved the random_name for this model
    random_name = pipeline_step.model
    # Load the Keras model here
    keras_file_name = file_name[:-5] + random_name + '_keras_deep_learning_model.h5'

    model = keras.models.load_model(keras_file_name)
    # Put the model back in place so that we can still use it to get predictions without having to load it back in from disk
    return model

def load_dl_model(file_name):
    # file_name = "..\\mlweb\\trained_pipeline\\deepl\\auto_ml_saved_pipeline.dill"
    with open(file_name, 'rb') as read_file:
        base_pipeline = dill.load(read_file)

    for step in base_pipeline.named_steps:
        pipeline_step = base_pipeline.named_steps[step]
        if pipeline_step.get('model_name', 'reallylongnonsensicalstring')[:12] == 'DeepLearning':
            print("DeepLearning True")
            print(pipeline_step)
            pipeline_step.model = insert_deep_learning_model(pipeline_step, file_name)
        else:
            print("long name")
    return base_pipeline

class TestD(object):
    def __init__(self, **kwargs):
        super(TestD, self).__init__(**kwargs)
        self.trained_ml_pipeline = Sequential()
        # self.trained_ml_pipeline.load_weights("..\\mlweb\\trained_pipeline\\deepl\\deep_learning.h5")
        # self.trained_ml_pipeline = keras.models.load_model("..\\mlweb\\trained_pipeline\\deepl\\0.41750410473762245_keras_deep_learning_model.h5")
        # self.trained_ml_pipeline = load_dl_model("..\\mlweb\\trained_pipeline\\deepl\\auto_ml_saved_pipeline.dill")
        # self.trained_ml_pipeline = keras.models.load_model("..\\mlweb\\trained_pipeline\\deepl\\deep_learning.h5")

    # def test_deepl(self, test_data):
    #     Predictions = self.trained_ml_pipeline.predict(test_data)
    #     return Predictions

    # def dl_prediction(self):
    #     predictions = self.trained_ml_pipeline.predict(np.loadtxt(strpath, delimiter=","))
    #     print(predictions.shape)
    #     my_predictions=self.trained_ml_pipeline.predict(predictions)
    #     return my_predictions

    def load_dl_weights(self, test_X):
        n_cols = test_X.shape[1]
        #add model layers
        self.trained_ml_pipeline.add(Dense(10, activation='relu', input_shape=(n_cols,)))
        self.trained_ml_pipeline.add(Dense(10, activation='relu'))
        self.trained_ml_pipeline.add(Dense(1))
        self.trained_ml_pipeline.compile(optimizer='adam', loss='mean_squared_error')        
        self.trained_ml_pipeline.load_weights("..\\mlweb\\trained_pipeline\\deepl\\deep_learning.h5")

    def dl_predict(self, test_X):
        self.load_dl_weights(test_X)
        test_y_predictions = self.trained_ml_pipeline.predict(test_X)
        test_y_predictions = (test_y_predictions > 0.5)
        return test_y_predictions

    def dl_evaluate(self, test_X, test_Y):
        self.trained_ml_pipeline.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        score, acc = self.trained_ml_pipeline.evaluate(test_X, test_Y)
        return score, acc

# print(file_name[:-5])
file_name = "..\\mlweb\\trained_pipeline\\deepl\\auto_ml_saved_pipeline.dill"
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

class AutoLearn:
    def __init__(self):
        # self.trained_ml_pipeline = Sequential()
        self.trained_ml_pipeline = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
        with open(file_name, 'rb') as read_file:
            datastruct = dill.load(read_file)
        for step in datastruct.named_steps:
            pipeline_step = datastruct.named_steps[step]
            if pipeline_step.get('model_name', 'reallylongnonsensicalstring')[:12] == 'DeepLearning':
                keras_file_name = file_name[:-5] + pipeline_step.model + '_keras_deep_learning_model.h5'
                print(keras_file_name)
                self.trained_ml_pipeline = load_model(keras_file_name)       

    def test_deepl(self, test_data):
        Predictions = self.trained_ml_pipeline.predict(test_data)
        return Predictions

    def getsummary(self):
        return self.trained_ml_pipeline.summary()

##Test
# df_test = {'step':'0', 'customer':'1959', 'age':'7', 'gender':'0', 'zipcodeOri':'0', 'merchant':'46', 'zipMerchant':'0', 'category':'3', 'amount':'4.55'}
# # df_Y = {'fraud':'0'}
# al = AutoLearn()
# al.getsummary()
# al.test_deepl(df_test)

class MLEngine (Test, TestD):
    def __init__(self, **kwargs):
        super(MLEngine, self).__init__(**kwargs)
        # Test.__init__(self)
        # TestD.__init__(self)
        self.threshold = 0.5

    def handle_non_numerical_data(self, test_X):
        columns = test_X.columns.values
        for column in columns:
            text_digit_vals = {}
            def convert_to_int(val):
                return text_digit_vals[val]

            if test_X[column].dtype != np.int64 and test_X[column].dtype != np.float64:
                column_contents = test_X[column].values.tolist()
                unique_elements = set(column_contents)
                x = 0
                for unique in unique_elements:
                    if unique not in text_digit_vals:
                        text_digit_vals[unique] = x
                        x+=1
                test_X[column] = list(map(convert_to_int, test_X[column]))
    
    def predict(self, test_X):
        y_pred = self.test_forest(test_X)
        score = self.getScore(test_X)
        print("SCore -->", score)
        if(score > 0.4 and score < 0.6):
            self.handle_non_numerical_data(test_X)
            y_pred = self.dl_predict(pd.DataFrame(test_X, index=[0]))
        y_pred = (y_pred > 0.5)
        return y_pred


# df_test2 = {'step':'157', 'customer':'C583110837', 'age':'3', 'gender':'M', 'zipcodeOri':'28807', 'merchant':'M480139044', 'zipMerchant':'28807', 'category':'es_health', 'amount':'16.26'}
# engine = MLEngine()
# pred = engine.predict(df_test2)
