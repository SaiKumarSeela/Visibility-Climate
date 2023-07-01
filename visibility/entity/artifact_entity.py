from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    r2_score: float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_r2_score: float
    best_model_r2_score: float
   

@dataclass
class ModelPusherArtifact:
    saved_model_path: str
    model_file_path: str
