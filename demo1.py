import random
# utils contains the custom datasets and other miscellaneous functions for data processing and evaluation
from utils import *
train_set = Multimodal_Datasets(data_path, "mosei", "train")
valid_set = Multimodal_Datasets(data_path, "mosei", "valid")
test_set = Multimodal_Datasets(data_path, "mosei", "test")

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
              "source": None, ["text", "video"] # state the source(s)
              "target": None, ["audio"]  # state the target
              "missing_rate": 1, # configure missing rate to adjust how frequent the test set contains incomplte samples
              }

# get the training, valid. data based on the required modalities
get_train_valid_data(exp_config)
# get the source and target dataset(s) for gen. model
get_src_tgt(exp_config)

# get the test data
get_test_data(exp_config)
# get the source dataset for feature simulation
get_sim_src_tgt(exp_config)

# generative model config.
model_config = {
    "batch_size": random.choice([32,64]),
    "lr": random.choice([0.0001, 0.0005, 0.001]),
    "hidden_size": random.choice([64, 128, 256]),
    "dropout": random.choice([0.2, 0.5, 0.8]),
    "epochs": 100,
    "enforce_rate": random.choice([0.2, 0.5, 0.7])
    }
# prediction model config.
gen_model_config = {
    "batch_size": random.choice([32, 64]),
    "lr": random.choice([0.0001 ,0.0005, 0.001]),
    "hidden_size": random.choice([64, 128, 256, 512]),
    "dropout": random.choice([0.2, 0.5, 0.7]),
    "epochs": 150
    }
	
# train the generative model
train_gen(exp_config, gen_model_config, "text->video")
# simulate the missing features
sim_missing_features(exp_config, "text->video")

# train the prediction model and evaluate results
train_ef(exp_config, model_config)
