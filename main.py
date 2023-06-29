from visibility.pipelines.training_pipeline import TrainPipeline
from visibility.logger import logging


if __name__ == "__main__":
    logging.info("Pipeline started")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
    logging.info("Pipeline run Successfully")