# Multimodal-Sentiment-Anlaysis-with-Missing-Modality-Generation


An example usecase of training a tri-modal model with early fusion on MOSI:

```python


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
              
              # for get_train_valid_data
              "X_train": None,
              "y_train": None,
              "X_valid": None,
              "y_valid": None,
              
              # for get_test_data
              "X_test": None,
              "y_test": None,
              "tokens": None,
              
<!--               # for gen. model 
              "source": None,
              "target": None,
              "missing_rate": 0.5,
              
              # for get_src_tgt
              "src_train": {},
              "tgt_train": {},
              "src_valid": {},
              "tgt_valid": {},
              
              # for simulate_missing_features
              "src_sim": {}, 
              "ref_idx": {},
              "X_test_m": None,
              "X_test_s": None,
              # Models
              "Generative": {} -->
              }

get_train_valid_data(exp_config)
get_test_data(exp_config)




```
