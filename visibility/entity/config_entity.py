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
class DataValidationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.data_validation_dir: str = os.path.join(self.training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,)
    
class DataTransformationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir,
            training_pipeline.TRAIN_FILE_NAME.replace("csv","npy"))
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,
            training_pipeline.TEST_FILE_NAME.replace("csv","npy"))
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)
        
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
 