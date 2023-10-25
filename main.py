from kdcnnClassifier import logger
from kdcnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from kdcnnClassifier.pipeline.stage_02_model_training import ModelTrainingPipeline

'''
STAGE_NAME = "STAGE 01 : Data Ingestion"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e
'''

STAGE_NAME = "STAGE 02 : Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
