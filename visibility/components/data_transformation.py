import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


from visibility.constants.training_pipeline import TARGET_COLUMN,DATE_COLUMN
from visibility.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from visibility.constants.training_pipeline import SCHEMA_FILE_PATH

from visibility.entity.config_entity import DataTransformationConfig
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
from visibility.utils.main_utils import save_numpy_array_data,save_object,read_yaml_file

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transfromation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise VisibilityClimateException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VisibilityClimateException(e, sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:

            standard_scaler = StandardScaler()
            
            imputer = KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            preprocessor = Pipeline(
                steps= [
                    ("imputer",imputer),
                    ("StandardScaler",standard_scaler) #keep every feature in same range 
                ]
            )
            return preprocessor
        except Exception as e:
            raise VisibilityClimateException(e, sys) from e
    def replaceInvalidValuesWithNull(self,data):

        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data
    
    def drop_columns_after_preprocessing(self,features_df):
        drop_columns = self._schema_config['drop_columns']
        features_df = features_df.drop(drop_columns, axis = 1)
        return features_df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiate data transformation and get data transformed objects")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            train_df = self.replaceInvalidValuesWithNull(train_df)
            test_df = self.replaceInvalidValuesWithNull(test_df)

            preprocessor = self.get_data_transformer_object()
            
            logging.info("Separate the dataframe into input features and target feature")
            #training dataframe 
            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN,DATE_COLUMN],axis =1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            

            #testing dataframe 
            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN,DATE_COLUMN],axis =1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # After preprocessing and analyzing correlations b/w features we should drop some of the columns
            logging.info("Drop some of the columns after analysis")
            input_feature_train_df = self.drop_columns_after_preprocessing(input_feature_train_df)
            input_feature_test_df = self.drop_columns_after_preprocessing(input_feature_test_df)

            logging.info("Start feature scaling")
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_features  = preprocessor_object.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_features, np.array(target_feature_test_df) ]

            logging.info("Save the data into numpy arrays")
            #save numpy array data
            save_numpy_array_data(self.data_transfromation_config.transformed_train_file_path,array= train_arr,)
            save_numpy_array_data(self.data_transfromation_config.transformed_test_file_path,array = test_arr,)

            save_object(self.data_transfromation_config.transformed_object_file_path,preprocessor,)

            #preparing artifact
            data_transfomation_artifact = DataTransformationArtifact(
                transformed_object_file_path= self.data_transfromation_config.transformed_object_file_path,
                transformed_train_file_path= self.data_transfromation_config.transformed_train_file_path,
                transformed_test_file_path= self.data_transfromation_config.transformed_test_file_path,
            )
            logging.info("Data transformation artifact :{data_transformation_artifact}")
            return data_transfomation_artifact



        except Exception as e:
            raise VisibilityClimateException(e,sys)
