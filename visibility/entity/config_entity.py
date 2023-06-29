from datetime import datetime
import os
from visibility.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%y_%H_%M_%S")
        self.is_pipeline_running = False
        self.timetamp = timestamp
        self.artifact_dir = os.path.join(training_pipeline.ARTIFACT_NAME,timestamp)
class DataIngestionConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.data_ingestion_dir: str = os.path.join(self.training_pipeline_config.artifact_dir,
                                                    training_pipeline.DATA_INGESTION_DIR)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir,
                                                 training_pipeline.FEATURE_STORE_FILE_PATH,
                                                 training_pipeline.RAW_FILE_NAME)
        self.train_file_path:str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTED_PATH,
                                                training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path: str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTED_PATH,
                                                training_pipeline.TEST_FILE_NAME)


        