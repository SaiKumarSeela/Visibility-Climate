from visibility.entity.config_entity import DataIngestionConfig
from visibility.entity.artifact_entity import DataIngestionArtifact
from visibility.data_access.climate_data import ClimateData
from visibility.logger import logging
from visibility.constants.training_pipeline import TRAIN_TEST_SPLIT_RATIO
from sklearn.model_selection import train_test_split
import os
import pandas as pd
class DataIngestion:
    def __init__(self,dataingestion_config:DataIngestionConfig):
        
        self.dataingestion_config = dataingestion_config
    def get_data_from_database(self):
        logging.info("Get the data from database")
        self.feature_store_file_path = self.dataingestion_config.feature_store_file_path

        cd = ClimateData()
        dataframe =cd.extract_data_from_db()

        dir = os.path.dirname(self.feature_store_file_path)
        os.makedirs(dir,exist_ok=True)

        dataframe.to_csv(self.feature_store_file_path,index = False)
        return dataframe
    def split_into_train_and_test(self,dataframe:pd.DataFrame):
        logging.info("Split the data into train and test")
        split_ratio = TRAIN_TEST_SPLIT_RATIO
        train_df, test_df = train_test_split(dataframe,test_size=split_ratio)

        test_file_path = self.dataingestion_config.test_file_path
        train_file_path = self.dataingestion_config.train_file_path

        test_dir = os.path.dirname(test_file_path)
        train_dir = os.path.dirname(train_file_path)
        os.makedirs(test_dir,exist_ok=True)

        test_df.to_csv(test_file_path,index=False)
        train_df.to_csv(train_file_path,index=False)

        return train_file_path,test_file_path
    def initiate_data_ingestion(self):
        logging.info("Initiate data ingestion")

        df = self.get_data_from_database()
        train_file_path,test_file_path = self.split_into_train_and_test(df)
        return DataIngestionArtifact(
            feature_store_file_path= self.feature_store_file_path,
            train_file_path=train_file_path,
            test_file_path=test_file_path
        )