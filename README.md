# Multimodal-Sentiment-Anlaysis-with-Missing-Modality-Generation


An example usecase of training a tri-modal model with early fusion:

```python

from Models.Generative import *
from Models.LateFusion import *
from Models.EarlyFusion import *
from Models.MemoryFusion import *



exp_config = {# define below to start
              "dataset_name": train_set.data,
              "train": train_set,
              "valid": valid_set,
              "test": test_set,
              "modalities": ["text"],
              # for get_train_valid_data
              "X_train": None,
              "y_train": None,
              "X_valid": None,
              "y_valid": None,
              # for get_test_data
              "X_test": None,
              "y_test": None,
              "tokens": None,
              # for autoencoder 
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
              "Autoencoder": {}}

get_train_valid_data(exp_config)
get_test_data(exp_config)





```
