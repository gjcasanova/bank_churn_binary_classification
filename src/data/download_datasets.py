"""Download and extract the required datasets."""

# Native
import argparse
import os
from zipfile import ZipFile

# Third party
import pandas as pd
from kaggle import KaggleApi
from sklearn.model_selection import train_test_split

# Local
from src.config.settings import COMPETITION_NAME, PROJECT_DATA_PATH

EXTRACTION_DESTINY_PATH = PROJECT_DATA_PATH / 'raw'
DONWLOAD_DESTINY_PATH = PROJECT_DATA_PATH / 'raw'
RESEARCH_DATA_PATH = PROJECT_DATA_PATH / 'raw' / 'research'
DOWNLOADED_COMPRESSED_FILE_PATH = DONWLOAD_DESTINY_PATH / f'{COMPETITION_NAME}.zip'
TRAIN_DATASET_FILE_PATH = EXTRACTION_DESTINY_PATH / 'train.csv'


def fetch_competition_dataset(competition_name, download_destiny_path):
    """Fetch and save a dataset from a remote origin."""
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kaggle_api.competition_download_files(competition_name, path=download_destiny_path, force=True)


def extract_files(compressed_file_path, extraction_destiny_path):
    """Exrtract content from a compressed file."""
    with ZipFile(compressed_file_path, 'r') as zip_file:
        zip_file.extractall(extraction_destiny_path)


def remove_file(file_path):
    """Remove a file."""
    os.remove(file_path)


def split_train_test_dataset(dataset_file_path, destiny_path, y_attribute_name):
    """Split a dataset into training and test."""
    dataset = pd.read_csv(dataset_file_path)
    X = dataset.drop([y_attribute_name], axis=1)
    y = dataset[[y_attribute_name]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)

    train_dataset = pd.concat([X_train, y_train], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)

    destiny_path.mkdir(parents=True, exist_ok=True)

    train_dataset.to_csv(destiny_path / 'train.csv', index=False)
    test_dataset.to_csv(destiny_path / 'test.csv', index=False)


def main(download_compressed=False, extract_compressed=False, remove_download=False, split_train_test=False):
    """Execute the flow."""
    if download_compressed:
        fetch_competition_dataset(COMPETITION_NAME, DONWLOAD_DESTINY_PATH)

    if extract_compressed:
        extract_files(DOWNLOADED_COMPRESSED_FILE_PATH, EXTRACTION_DESTINY_PATH)

    if remove_download:
        remove_file(DOWNLOADED_COMPRESSED_FILE_PATH)

    if split_train_test:
        split_train_test_dataset(TRAIN_DATASET_FILE_PATH, RESEARCH_DATA_PATH, 'Exited')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Competition files download helper')

    args_parser.add_argument('--download-compressed', action='store_true')
    args_parser.add_argument('--extract-compressed', action='store_true')
    args_parser.add_argument('--remove-download', action='store_true')
    args_parser.add_argument('--split-train-test', action='store_true')

    kwargs = vars(args_parser.parse_args())

    main(**kwargs)
