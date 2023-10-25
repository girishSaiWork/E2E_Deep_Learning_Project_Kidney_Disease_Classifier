from kdcnnClassifier.config.configuration import ConfigurationManager
from kdcnnClassifier.components.model_training import Training
from kdcnnClassifier import logger
import numpy as np

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        # training.data_splitter()
        # training.eval_folders()
        train_data_set, val_data_set, test_data_set = training.train_valid_generator()
        predictions, history = training.model_creation(train_data_set, val_data_set, test_data_set)
        predicted_values = [np.argmax(i) for i in predictions]
        actual_values = test_data_set.classes
        rows_in_dataset = actual_values.shape[0]
        accuracy = (predicted_values == actual_values).sum() / rows_in_dataset
        print(f'Accuracy : {accuracy}')


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
