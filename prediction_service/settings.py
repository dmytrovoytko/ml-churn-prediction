# show extra information for checking execution
DEBUG = False  # True # False

DATA_DIR = f'./data/'  # ! with '/' at the end!
MODEL_DIR = f'./model/'  # ! with '/' at the end!
MODEL_NAME = 'best_model.pkl'

# preprocessing, training, prediction
TARGET = "churn"  # labels column, should be lowercased, space -> '_'

# preprocessing
REMOVE_OUTLIERS = False  # True # False
USE_ENCODER = False

# web app settings
PORT = 5555

