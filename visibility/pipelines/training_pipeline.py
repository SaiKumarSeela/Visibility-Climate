from visibility.entity.config_entity import (TrainingPipelineConfig,DataIngestionConfig,
                                    DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig)
from visibility.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact,
                                               DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact)
from visibility.components.data_ingestion import DataIngestion
from visibility.components.data_validation import DataValidation
from visibility.components.data_transformation import DataTransformation
from visibility.components.model_trainer import ModelTrainer
from visibility.components.model_evaluation import ModelEvaluation
from visibility.components.model_pusher import ModelPusher
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
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info("Start data validation")
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validataion= DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=self.data_validation_config )
            data_validation_artifact = data_validataion.initiate_data_validate()
            return data_validation_artifact

        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("Start data transformation")
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact= data_validation_artifact,
                                                     data_transformation_config=self.data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact

        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config,
                                         data_transformation_artifact= data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def start_model_evaluation(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_artifact:ModelTrainerArtifact):
        try:
            self.model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                               data_transformation_artifact=data_transformation_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config= self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config= model_pusher_config,
                                            model_evaluation_artifact= model_evaluation_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model pusher completed: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise VisibilityClimateException(e,sys)
    def run_pipeline(self):
        try:
            logging.info("Start the Trainig Pipeline")
            
            self.is_pipeline_running = True
            if self.is_pipeline_running:
                dataingestion_artifact =self.start_data_ingestion()
                logging.info(f"Completed data ingestion and get the artifact {dataingestion_artifact}")
                datavalidation_artifact = self.start_data_validation(data_ingestion_artifact=dataingestion_artifact)
                logging.info(f"Completed data validation and get the artifact {datavalidation_artifact}")
                datatransformation_artifact = self.start_data_transformation(data_validation_artifact= datavalidation_artifact)
                logging.info(f"Completed data transformation and get the artifact {datatransformation_artifact}")
                modeltrainer_artifact = self.start_model_trainer(data_transformation_artifact=datatransformation_artifact)
                logging.info(f"Completed model trainer and get the artifact {modeltrainer_artifact}")
                modelevaluation_artifact= self.start_model_evaluation(data_transformation_artifact=datatransformation_artifact,
                                                                      model_trainer_artifact=modeltrainer_artifact)
                logging.info(f"Completed model evaluation and get the artifact {modelevaluation_artifact}")
                modelpusher_artifact = self.start_model_pusher(model_evaluation_artifact=modelevaluation_artifact)
                logging.info(f"Completed model pusher and get the artifact {modelpusher_artifact}")

        except Exception as e:
            raise VisibilityClimateException(e,sys)
