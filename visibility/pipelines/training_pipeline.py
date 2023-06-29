from visibility.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from visibility.entity.artifact_entity import DataIngestionArtifact
from visibility.components.data_ingestion import DataIngestion
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
import sys
class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.is_pipeline_running: bool = False
    def start_data_ingestion(self):
        try:
            logging.info("Start data ingestion")
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        
        except Exception as e:
            raise VisibilityClimateException(e,sys)

    def run_pipeline(self):
        try:
            logging.info("Start the Trainig Pipeline")
            
            self.is_pipeline_running = True
            if self.is_pipeline_running:
                dataingestion_artifact =self.start_data_ingestion()
                logging.info(f"Completed data ingestion and get the artifact {dataingestion_artifact}")
        except Exception as e:
            raise VisibilityClimateException(e,sys)
