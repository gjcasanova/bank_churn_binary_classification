"""Save settings for the project like constants and other general values."""

# Native
from pathlib import Path

# Paths
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
PROJECT_DATA_PATH = PROJECT_ROOT_PATH / 'data'

# Remote sources
COMPETITION_NAME = 'playground-series-s4e1'
