import numpy as np
import pickle
import os
import random

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data.dataset import Dataset
    
class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data, split_type):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data +'.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META


def get_src_tgt(config):
    
    assert len(config["target"]) > 0, "Specify which modality(ies) are missing."
    src = config["source"]
    tgt = config["target"]
    
    def my_collate_fn(item_list):
        _src, _tgt = {s: [] for s in src}, {t: [] for t in tgt}
        for item in item_list:
            x, y, _ = item
            _, x_t, x_a, x_v = x
            dummy_dict = {"text": x_t, "audio": x_a, "video": x_v}
            for s in src:
                _src[s].append(dummy_dict[s])
            for t in tgt:
                _tgt[t].append(dummy_dict[t])

        for s in src:
            _src[s] = torch.stack(_src[s])
        for t in tgt:
            _tgt[t] = torch.stack(_tgt[t])

        return _src, _tgt

    for tv in ["train", "valid"]:
        dataset = config[tv]
        N = len(dataset)
        for batch in DataLoader(dataset, batch_size = N, collate_fn = my_collate_fn):
            _src, _tgt = batch
        
        for s in src:
            config[f"src_{tv}"][s] = _src[s]
        for t in tgt:
            config[f"tgt_{tv}"][t] = _tgt[t]


def get_sim_src_tgt(config):
    
    assert len(config["target"]) > 0, "Specify which modality(ies) are missing."
    src = config["source"]
    tgt = config["target"]
    tokens = config["tokens"]

    def my_collate_fn(item_list):
        _src = {s: [] for s in src}
        for item in item_list:
            x, y, _ = item
            _, x_t, x_a, x_v = x
            dummy_dict = {"text": x_t, "audio": x_a, "video": x_v}
            for s in src:
                _src[s].append(dummy_dict[s])

        for s in src:
            _src[s] = torch.stack(_src[s])

        return _src

    dataset = config["test"]
    N = len(dataset)
    for batch in DataLoader(dataset, batch_size = N, collate_fn = my_collate_fn):
        _src = batch
    
    for s in src:
        config["src_sim"][s] = _src[s][tokens]

def get_train_valid_data(config):

    modalities = config["modalities"]

    def my_collate_fn(item_list):
        X, Y, ref_idx = [], [], {}
        for item in item_list:
            x, y, _ = item
            _, x_t, x_a, x_v = x
            dummy_dict = {"text": x_t, "audio": x_a, "video": x_v}
            X.append( torch.cat([dummy_dict[m] for m in modalities], 1) )
            Y.append(y)

        start = 0
        for m in modalities:
            end = start + dummy_dict[m].shape[1]
            ref_idx[m] = [start, end]
            start = end

        return torch.stack(X), torch.stack(Y).squeeze(), ref_idx

    for tv in ["train", "valid"]:
        dataset = config[tv]
        N = len(dataset)
        for batch in DataLoader(dataset, batch_size = N, collate_fn = my_collate_fn):
            config[f"X_{tv}"], config[f"y_{tv}"], config["ref_idx"] = batch

            if config["dataset_name"] == "iemocap":
                config[f"y_{tv}"] = config[f"y_{tv}"].argmax(1)


def get_test_data(config):

    modalities = config["modalities"]
    missing = config["target"]
    missing_rate = config["missing_rate"]

    def my_collate_fn(item_list):
        X, Y = [], []


        X_m, idx = None, None
        if missing:
            X_m, idx = [], []

        for item in item_list:
            x, y, _ = item
            i, x_t, x_a, x_v = x
            dummy_dict = {"text": x_t, "audio": x_a, "video": x_v}
            X.append( torch.cat([dummy_dict[m] for m in modalities], 1) )
            Y.append(y)
            if missing:
                if random.random() < missing_rate:
                    idx.append(i)
                    for miss in missing:
                        dummy_dict[miss] = torch.zeros_like(dummy_dict[miss])
                X_m.append( torch.cat([dummy_dict[m] for m in modalities], 1) )
        if missing:
            return torch.stack(X), torch.stack(X_m), torch.stack(Y).squeeze(), idx
        else:
            return torch.stack(X), None, torch.stack(Y).squeeze(), None

    dataset = config["test"]
    N = len(dataset)
    for batch in DataLoader(dataset, batch_size = N, collate_fn = my_collate_fn):
        config["X_test"], config["X_test_m"], config["y_test"], config["tokens"] = batch
        if config["dataset_name"] == "iemocap":
            config[f"y_test"] = config[f"y_test"].argmax(1)
        

def sim_missing_features(config, src_to_tgt):
    src_modality, tgt_modality = src_to_tgt.split("->")
    tokens = config["tokens"]
    src_sim = config["src_sim"][src_modality]
    start, end = config["ref_idx"][tgt_modality]
    X_test = config["X_test"]
    autoencoder = config["Autoencoder"][src_to_tgt]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        N, T = src_sim.shape[:2]
        D = config["tgt_valid"][tgt_modality].shape[2]
        autoencoder.eval()
        tgt_sim = autoencoder(src_sim.to(device), torch.zeros((N, T, D)).to(device))

    X_test_s = config["X_test_m"].clone()
    X_test_s[tokens, :, start:end] = tgt_sim.cpu()
    config["X_test_s"] = X_test_s


def evaluate_metrics(dataset, predictions, y_test):
    if dataset in ["mosi", "mosei"]:
        y_test = y_test.cpu().data.numpy()
        predictions = predictions.cpu().data.numpy()

        non_zeros = (y_test != 0.0)

        mae = np.mean(np.absolute(predictions - y_test)) 
        print("MAE:", mae)

        corr = np.corrcoef(predictions,y_test)[0][1]
        print("Corr :", corr)

        m7_predictions = np.clip(predictions, a_min=-3., a_max=3.)
        mult = round(sum(np.round(m7_predictions)==np.round(y_test))/ float(len(y_test)),5)
        print("M7:", mult)
    
        f1 = f1_score((predictions[non_zeros] > 0), (y_test[non_zeros] > 0), average='weighted')
        print("F1:", f1)

        binary_truth = (y_test[non_zeros] > 0)
        binary_preds = (predictions[non_zeros] > 0)
        
        ba = accuracy_score(binary_truth, binary_preds)
        print("Binary Ac:", ba)
        print("Confusion Matrix:")
        print(confusion_matrix(binary_truth, binary_preds))

#     elif dataset == "iemocap":
#         emos = ["Neutral", "Happy", "Sad", "Angry"]
#         y_test = y_test.cpu().data.numpy()
#         predictions = F.softmax(predictions, 1).argmax(1).cpu().data.numpy()

#         print('-' * 50)
#         for i in range(4):
#             print(emos[i])
#             _y_test = y_test[y_test == i] == i
#             _predictions = predictions[y_test == i] == i
#             f1 = f1_score(_y_test, _predictions, "weighted")
#             acc = accuracy_score(_y_test, _predictions)
#             print("F1:", f1)
#             print("Acc:", acc)
#             print('-' * 50)
            
def return_missing_features(config, src_to_tgt):
    src_modality, tgt_modality = src_to_tgt.split("->")
    tokens = config["tokens"]
    src_sim = config["src_sim"][src_modality]
    start, end = config["ref_idx"][tgt_modality]
    X_test = config["X_test"]
    autoencoder = config["Autoencoder"][src_to_tgt]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        N, T = src_sim.shape[:2]
        D = config["tgt_valid"][tgt_modality].shape[2]
        autoencoder.eval()
        tgt_sim = autoencoder(src_sim.to(device), torch.zeros((N, T, D)).to(device))

    return tgt_sim
