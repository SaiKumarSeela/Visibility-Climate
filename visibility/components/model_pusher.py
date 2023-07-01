from visibility.utils.main_utils import load_numpy_array_data
from visibility.exception import VisibilityClimateException
from visibility.logger import logging
from visibility.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from visibility.entity.config_entity import ModelTrainerConfig,ModelPusherConfig
import os,sys,shutil
from sklearn.metrics import r2_score
from visibility.ml.model.estimator import VisibilityClimateModel
from visibility.utils.main_utils import save_object,load_object

class ModelPusher:

    def __init__(self,model_pusher_config: ModelPusherConfig,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise VisibilityClimateException(e,sys)

    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path

            #creating model pusher dir to save model,this model is for training purpose
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok= True)
            shutil.copy(src = trained_model_path,dst = model_file_path)

            #saved model dir , this model is for production purpose
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok= True)
            shutil.copy(src = trained_model_path,dst= saved_model_path)

            #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path,model_file_path=model_file_path)

            return model_pusher_artifact

        except Exception as e:
            raise VisibilityClimateException(e,sys)

