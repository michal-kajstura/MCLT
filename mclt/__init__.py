from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
DATA_PATH = PROJECT_PATH / 'data'
DATASETS_PATH = DATA_PATH / 'datasets'
MODEL_DIR = DATA_PATH / 'models'
# DEFAULT_MLFLOW_TRACKING_URI = str(PROJECT_PATH / 'mlruns')
DEFAULT_MLFLOW_TRACKING_URI = '/home/michal/studia/logs/mlruns'
