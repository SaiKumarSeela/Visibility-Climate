from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from xgboost import XGBRegressor
from visibility.logger import logging
from visibility.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
from visibility.exception import VisibilityClimateException
import sys,os
import warnings

warnings.filterwarnings("ignore")


class Model_Finder:
    """
    This class shall  be used to find the model with best accuracy and AUC score.      
    """

    def __init__(self):
        
        self.clf = RandomForestClassifier()
        self.DecisionTreeReg = DecisionTreeRegressor()
    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception
            """
        logging.info('Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            logging.info('Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            logging.info('Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            logging.info('Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise VisibilityClimateException(e,sys)
    def get_best_params_for_DecisionTreeRegressor(self, train_x, train_y):
        """
                Method Name: get_best_params_for_DecisionTreeRegressor
                Description: get the parameters for DecisionTreeRegressor Algorithm which give the best accuracy.
                                Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception
        """
        logging.info('Entered the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_decisionTree = {"criterion": ["mse"],
                              "splitter": ["best", "random"],
                              "max_features": ["auto", "sqrt"],
                              'max_depth': range(2, 10, 2),
                              'min_samples_split': range(2, 10, 2)
                              }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.DecisionTreeReg, self.param_grid_decisionTree, verbose=3,cv=3)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.max_features = self.grid.best_params_['max_features']
            self.max_depth  = self.grid.best_params_['max_depth']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
          
            self.decisionTreeReg = DecisionTreeRegressor(criterion=self.criterion,splitter=self.splitter,max_features=self.max_features,max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            logging.info('Decision Tree best params: ' + str(
                                       self.grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return self.decisionTreeReg
        except Exception as e:
            logging.info('Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            logging.info('knn Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise VisibilityClimateException(e,sys)

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                            Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception

        """
        logging.info('Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5, 10],
                'n_estimators': [10, 50, 100]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBRegressor(objective='reg:linear'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBRegressor(objective='reg:linear',learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            logging.info('XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            logging.info('Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            logging.info('XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise VisibilityClimateException(e,sys)


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
            Method Name: get_best_model
            Description: Find out the Model which has the best AUC score.
            Output: The best model name and the model object
            On Failure: Raise Exception

        """
        logging.info('Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:

            self.decisionTreeReg= self.get_best_params_for_DecisionTreeRegressor(train_x, train_y)
            self.prediction_decisionTreeReg = self.decisionTreeReg.predict(test_x) # Predictions using the decisionTreeReg Model
            self.decisionTreeReg_error = r2_score(test_y,self.prediction_decisionTreeReg)



         # create best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)  # Predictions using the XGBoost Model
            self.prediction_xgboost_error = r2_score(test_y,self.prediction_xgboost)


            #comparing the two models
            if(self.decisionTreeReg_error <  self.prediction_xgboost_error):
                return 'XGBoost',self.xgboost,self.prediction_xgboost_error
            else:
                return 'DecisionTreeReg',self.decisionTreeReg,self.decisionTreeReg_error

        except Exception as e:
            logging.info('Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            logging.info('Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise VisibilityClimateException(e,sys)

class VisibilityClimateModel:
    
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model =model
        except Exception as e:
            raise e

    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise e

class ModelResolver:
    # it is helping us to findout the best model
    def __init__(self,model_dir= SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def get_best_model_path(self) -> str:
        try:
            timestamp = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamp)
            latest_model_path = os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise VisibilityClimateException(e,sys)
        
    def is_model_exist(self)->str:
        try:
            if not os.path.exists(self.model_dir):
                return False
            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False
            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False
            return True
        except Exception as e:
            raise VisibilityClimateException(e,sys)