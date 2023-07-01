
from visibility.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from visibility.entity.config_entity import ModelTrainerConfig
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
from visibility.utils.main_utils import load_numpy_array_data,load_object,save_object
from visibility.ml.model.estimator import Model_Finder,VisibilityClimateModel,ModelResolver
import pandas as pd
import sys,os

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    

  
    def initiate_model_trainer(self):
        try:
            logging.info("Initiate model trainer")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path= self.data_transformation_artifact.transformed_test_file_path

            logging.info("Load numpy array data")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info("Try to find the best model out of xgboost and decision tree")
            model_finder = Model_Finder()
    
            best_model_name,best_model ,r2_score= model_finder.get_best_model(x_train,y_train,x_test,y_test)
            
            #The goal in regression is to minimize the difference between the predicted and actual values on unseen data,
            #  rather than comparing train and test scores directly.
            logging.info(f"Store the {best_model_name} model as object in directory")
            preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            visibility_climate_model = VisibilityClimateModel(preprocessor= preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj= visibility_climate_model)

            
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          r2_score=r2_score)


            logging.info("Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise VisibilityClimateException(e,sys)
        



