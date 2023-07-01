from visibility.entity.config_entity import ModelEvaluationConfig
from visibility.entity.artifact_entity import ModelTrainerArtifact,ModelEvaluationArtifact,DataTransformationArtifact
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
from visibility.constants.training_pipeline import TARGET_COLUMN,FEATURE_COLUMNS
from visibility.utils.main_utils import load_object,write_yaml_file,load_numpy_array_data
from visibility.ml.model.estimator import ModelResolver
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import sys



class ModelEvaluation:
    def __init__(self,model_evaluation_config:ModelEvaluationConfig, 
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    

    def initiate_model_evaluation(self):
        try:
            logging.info("Initiate model evaluation")
            train_file_path =  self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("load numpy data")
            #valid train and test file dataframe
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # df = pd.concat([train_arr,test_arr])
            # y_true = df[TARGET_COLUMN]
            # df.drop(TARGET_COLUMN,axis=1,inplace=True)
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("Concatenate arrat data and convert it into dataframe")
            feature_arr = np.concatenate((x_train, x_test))
            df = pd.DataFrame(feature_arr,columns=FEATURE_COLUMNS)

            target_arr = np.concatenate((y_train,y_test))
            y_true = pd.Series(target_arr)


            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True


            if not model_resolver.is_model_exist():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_r2_score=self.model_trainer_artifact.r2_score,
                    best_model_r2_score=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            logging.info("Get the best model")
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)
            
            y_trained_pred = train_model.predict(df) 
            y_latest_pred  =latest_model.predict(df) 
            logging.info("Get the r2 score on both trained and latest model")
            trained_score = r2_score(y_true, y_trained_pred)
            latest_score = r2_score(y_true, y_latest_pred)

            improved_accuracy = trained_score-latest_score
            if self.model_evaluation_config.change_threshold < improved_accuracy:
                
                is_model_accepted=True
            else:
                is_model_accepted=False

            
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_r2_score=trained_score,
                    best_model_r2_score=latest_score)
        
            model_eval_report = model_evaluation_artifact.__dict__

            #save the report
            logging.info("Save the report")
            write_yaml_file(self.model_evaluation_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise VisibilityClimateException(e,sys)
        