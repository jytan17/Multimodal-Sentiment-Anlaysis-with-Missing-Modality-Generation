
# utils contains the custom datasets and other miscellaneous functions for data processing and evaluation
from utils import *
train_set = Multimodal_Datasets(data_path, "mosi", "train")
valid_set = Multimodal_Datasets(data_path, "mosi", "valid")
test_set = Multimodal_Datasets(data_path, "mosi", "test")

# import models from "Models"
from Models.Generative import *
from Models.LateFusion import *
from Models.EarlyFusion import *
from Models.MemoryFusion import *


# create a dictionaty to carry all the necessary items for training
exp_config = {# define below to start
              "dataset_name": train_set.data,
              "train": train_set,
              "valid": valid_set,
              "test": test_set,
              "modalities": ["text", "audio", "video"], # define the modalities to incoporate
              
              # The following are required only when training a genrative model to simulate missing modalities
              "source": None, # state the source
              "target": None, # state the target
              "missing_rate": 1, # configure missing rate to adjust how frequent the test set contains incomplte samples
              }

# get the training, valid. and test data based on the required modalities
get_train_valid_data(exp_config)
get_test_data(exp_config)

model_config = {
    "batch_size": 32,
    "lr": random.choice([0.0001 ,0.0005, 0.001]),
    "hidden_size": random.choice([64, 128, 256, 512]),
    "dropout": random.choice([0.2, 0.5, 0.7]),
    "epochs": 150}

train_ef(exp_config, model_config)
