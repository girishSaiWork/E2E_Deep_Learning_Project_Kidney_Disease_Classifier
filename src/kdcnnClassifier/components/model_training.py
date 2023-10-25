import os
from glob import glob
from pathlib import Path
import splitfolders
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from kdcnnClassifier import logger
from kdcnnClassifier.entity.config_entity import TrainingConfig


def imageCount(fPath):
    image_counts = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']
    imgList = []

    folders = os.listdir(fPath)
    folder_paths = [Path(fPath) / folder for folder in folders]

    for folder_path in folder_paths:
        image_count = sum(1 for item in folder_path.iterdir() if item.is_file() and item.suffix in image_extensions)
        image_counts[folder_path.name] = image_count
        imgList.append(image_count)

    imgSum = sum(imgList)
    return imgSum, imgList


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def data_splitter(self):
        try:

            data_dir = self.config.training_data
            train_ratio = self.config.params_train_ratio
            val_ratio = self.config.params_validation_ratio
            test_ratio = self.config.params_test_ratio
            output_data_dir = self.config.output_dir
            splitfolders.ratio(data_dir, seed=1337, output=output_data_dir, ratio=(train_ratio, val_ratio, test_ratio))
            logger.info(
                f"Data is splitted into TRAIN : {train_ratio}, VALIDATION : {val_ratio} and TEST : {test_ratio} ")

        except Exception as e:
            raise e

    def eval_folders(self):
        output_data_dir = self.config.output_dir
        testPath = output_data_dir / "test"
        trainPath = output_data_dir / "train"
        valPath = output_data_dir / "val"

        total_images, image_counts = imageCount(trainPath)
        print(f"Total images: {total_images}")
        for folder, count in zip(os.listdir(trainPath), image_counts):
            print(f"Number of images in '{folder}': {count}")

        total_images, image_counts = imageCount(valPath)
        print(f"Total images: {total_images}")
        for folder, count in zip(os.listdir(valPath), image_counts):
            print(f"Number of images in '{folder}': {count}")

        total_images, image_counts = imageCount(testPath)
        print(f"Total images: {total_images}")
        for folder, count in zip(os.listdir(trainPath), image_counts):
            print(f"Number of images in '{folder}': {count}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train_valid_generator(self):
        output_data_dir = self.config.output_dir
        testPath = output_data_dir / "test"
        trainPath = output_data_dir / "train"
        valPath = output_data_dir / "val"
        datagenerator_kwargs = dict(
            # rescale=self.config.params_rescale
            rescale=1 / 255
        )

        dataflow_kwargs = dict(
            target_size=(200, 200),
            # target_size=(self.config.params_img_h, self.config.params_img_w),
            color_mode=self.config.params_color_mode,
            class_mode=self.config.params_class_mode,
            batch_size=self.config.params_batch_size
        )

        train_datagen = ImageDataGenerator(**datagenerator_kwargs)
        valid_datagen = ImageDataGenerator(**datagenerator_kwargs)
        test_datagen = ImageDataGenerator(**datagenerator_kwargs)

        train_data_set = train_datagen.flow_from_directory(trainPath, **dataflow_kwargs)
        val_data_set = valid_datagen.flow_from_directory(valPath, **dataflow_kwargs)
        test_data_set = test_datagen.flow_from_directory(testPath, shuffle=False, **dataflow_kwargs)
        return train_data_set, val_data_set, test_data_set

    def model_creation(self, train_data_set, val_data_set, test_data_set):
        try:
            n_epochs = self.config.params_epochs
            train_input_shape = train_data_set.image_shape
            model = Sequential()

            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_input_shape))
            model.add(MaxPool2D(2))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPool2D(2))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPool2D(2))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPool2D(2))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPool2D(2))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPool2D(2))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(4, activation='softmax'))
            model.summary()

            METRICS = ['accuracy',
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall')
                       ]
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=METRICS)

            history = model.fit(
                train_data_set,
                validation_data=val_data_set,
                epochs=n_epochs
            )
            predictions = model.predict(test_data_set)
            model.save(self.config.trained_model_path)
            '''           
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            '''

            return predictions, history

        except Exception as e:
            logger.exception(e)
            raise e
