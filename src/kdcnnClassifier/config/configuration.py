from kdcnnClassifier.constants import *
from kdcnnClassifier.entity.config_entity import DataIngestionConfig, TrainingConfig
from kdcnnClassifier.utils.common import create_directories, read_yaml
import os


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        data_dir = self.config.data_ingestion.root_dir
        params = self.params
        # training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        training_data = self.config.data_ingestion.unzip_dir
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            output_dir=Path(training.output_dir),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_class_mode=params.CLASS_MODE,
            params_img_h=params.IMG_HEIGHT,
            params_img_w=params.IMG_WIDTH,
            params_train_ratio=params.TRAIN_RATIO,
            params_validation_ratio=params.VALIDATION_RATIO,
            params_test_ratio=params.TEST_RATIO,
            params_rescale=params.RESCALE,
            params_color_mode=params.COLOR_MODE,
            params_classes=params.CLASSES,
        )

        return training_config
