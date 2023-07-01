import pandas as pd   
from visibility.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from visibility.entity.config_entity import DataValidationConfig
from visibility.exception import VisibilityClimateException
from visibility.utils.main_utils import read_yaml_file,write_yaml_file
from visibility.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from visibility.logger import logging
import os,sys


class DataValidation:
    
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise VisibilityClimateException(e,sys)

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info("Required number of columns:{number_of_columns}")
            logging.info("Data Frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise VisibilityClimateException(e,sys)

    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            numerical_column_present = True
            missing_numerical_columns = []
            for numerical_column in numerical_columns:
                    if numerical_column not in dataframe.columns:
                        numerical_column_present =  False
                        missing_numerical_columns.append(numerical_column)
            if numerical_column_present:
                return True
            return False
        except Exception as e:
            raise VisibilityClimateException(e,sys)
            
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    
    def detect_dataset_drift(self,base_df,current_df,threshold = 0.5)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True 
                    status = False
                report.update({
                    column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status" : is_found
                    }
                })
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            #create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path= drift_report_file_path,content= report,)
            return status
        except Exception as e:
            raise VisibilityClimateException(e,sys)

    
    def initiate_data_validate(self)->DataValidationArtifact:
        try:
            error_message = ""
            logging.info("Initiate data validation")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Read the data from train and test")
            #Reading data from train and test file location 
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            #let's checke data drift

            logging.info("Detect the data drift")
            status = self.detect_dataset_drift(train_dataframe,test_dataframe)
            #validate number of columns

            logging.info("Validating no.of columns")
            status = self.validate_number_of_columns(dataframe= train_dataframe)
            if not status:
                error_message = f"{error_message} Train dataframe does not contaiin all columns"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain all columns"
            
            #check numerical columns 

            status = self.is_numerical_column_exist(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} Train dataframe does not contain all numercical columns"
            status = self.is_numerical_column_exist(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message} Test dataframe does not contain all numercical columns"
            
            if len(error_message)>0:
                raise Exception(error_message)
            
            
            
            logging.info("Return the datavalidation artifact")
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            return data_validation_artifact
        except Exception as e:
            raise VisibilityClimateException(e,sys)