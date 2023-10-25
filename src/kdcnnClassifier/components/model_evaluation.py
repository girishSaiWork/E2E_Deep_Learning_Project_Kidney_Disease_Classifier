import tensorflow as tf
from pathlib import Path
import os
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from kdcnnClassifier.entity.config_entity import EvaluationConfig
from kdcnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=(200, 200),
            # target_size=(self.config.params_img_h, self.config.params_img_w),
            color_mode=self.config.params_color_mode,
            class_mode=self.config.params_class_mode,
            batch_size=self.config.params_batch_size
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.trained_model_path)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        os.environ["MLFLOW_TRACKING_URI"] \
            = "https://dagshub.com/girishSaiWork/E2E_Deep_Learning_Project_Kidney_Disease_Classifier.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "girishSaiWork"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "069a4dd67a4255235d6151e96526830768fc5a80"
        """
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_uname
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_pwd
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="CNNModel")
            else:
                mlflow.keras.log_model(self.model, "model")
