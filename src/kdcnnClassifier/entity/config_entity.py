from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    output_dir: Path
    trained_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_color_mode: str
    params_img_h: int
    params_img_w: int
    params_train_ratio: float
    params_validation_ratio: float
    params_test_ratio: float
    params_rescale: float
    params_class_mode: str
    params_classes: int
