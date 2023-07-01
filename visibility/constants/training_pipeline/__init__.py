import os

SAVED_MODEL_DIR = os.path.join("saved_models")
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
RAW_FILE_NAME = "raw.csv"

TARGET_COLUMN = 'VISIBILITY'
DATE_COLUMN = 'DATE'
FEATURE_COLUMNS = ['DRYBULBTEMPF', 'RelativeHumidity', 'WindSpeed', 'WindDirection', 'SeaLevelPressure']
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml") #to give structure of the file
SHEMEA_DROP_COLS = "drop_columns"
#Training pipeline
ARTIFACT_NAME = "artifact"
"""
Constants related to DATA INGESTION 
"""
DATA_INGESTION_DIR = "data_ingestion"
FEATURE_STORE_FILE_PATH = "feature_store"
DATA_INGESTED_PATH = "ingested"
TRAIN_TEST_SPLIT_RATIO = 0.3

"""
Data validation related constants
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data transfromation related constants
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model Trainer related constants strat with MODEL TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

"""
Model Evaluation related constants start with MODEL EVALUATION VAR NAME
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_NAME: str = "report.yaml"

"""
Model pusher related constanst start with MODEL PUSHER
"""
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = SAVED_MODEL_DIR