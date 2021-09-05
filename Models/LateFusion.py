import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

save_path = os.getcwd()
save_path = os.path.join(save_path, "Models/SavedModels")

from importlib import reload 
import utils
utils = reload(utils)
evaluate_metrics = utils.evaluate_metrics

# from utils import evaluate_metrics


class LFLSTM(nn.Module):
    def __init__(self, modalities, input_sizes, hidden_sizes, output_size, dropout):
        super(LFLSTM, self).__init__()
        self.output_size = output_size
        

        self.lstmb, self.lstmc = None, None

        self.a, self.ha = input_sizes[0], hidden_sizes[0]
        self.lstma = nn.LSTM(self.a, self.ha, batch_first = True)

        if len(modalities) > 1:
            self.b, self.hb = input_sizes[1], hidden_sizes[1]
            self.lstmb = nn.LSTM(self.b, self.hb, batch_first = True)

        if len(modalities) > 2:
            self.c, self.hc = input_sizes[2], hidden_sizes[2]
            self.lstmc = nn.LSTM(self.c, self.hc, batch_first = True)


        self.fc1 = nn.Linear(sum(hidden_sizes), sum(hidden_sizes) // 2)
        self.fc2 = nn.Linear(sum(hidden_sizes) // 2, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):

        to_cat = []
        x_a = x[:,:,:self.a]
        x_a, _ = self.lstma(x_a)
        to_cat.append(x_a)
        if self.lstmb:
            x_b = x[:,:,self.a:self.a + self.b]
            x_b, _ = self.lstmb(x_b)
            to_cat.append(x_b)
        if self.lstmc:
            x_c = x[:,:,-self.c:]
            x_c, _ = self.lstmc(x_c)
            to_cat.append(x_c)

        x = torch.cat(to_cat, 2)

        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

def train_lf(config, model_config = None):

    def train(mode, X_train, y_train, loss_fn, batch_size):
        epoch_loss = 0
        model.train()
        N = X_train.shape[0]
        num_batches = 0
        loader = DataLoader(range(N), batch_size = batch_size, shuffle = False)
        for idx in loader:
            num_batches += 1
            optimizer.zero_grad()
            X_batch = X_train[idx].to(device); y_batch = y_train[idx].to(device)
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / num_batches
    
    def evaluate(model, X_valid, y_valid, loss_fn):
        model.eval()
        with torch.no_grad():
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            y_hat = model(X_valid)
            loss = loss_fn(y_hat, y_valid)
        return loss.item()

    def predict(model, X_test, y_test, loss_fn):
        model.eval()
        with torch.no_grad():
            X_test, y_test = X_test.to(device), y_test.to(device)
            predictions = model(X_test)
            loss = loss_fn(predictions, y_test)

        return predictions, loss.item()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train = config["X_train"]
    y_train = config["y_train"]
    X_valid = config["X_valid"]
    y_valid = config["y_valid"]
    X_test = config["X_test"]
    X_test_m = config["X_test_m"]
    X_test_s = config["X_test_s"]
    y_test = config["y_test"]
    modalities = config["modalities"]

    input_size = X_test.shape[2]
    output_size = 1

    models_config = []

    if model_config:
        models_config.append(model_config)
    else:
        for i in range(3):
            input_sizes = []
            hidden_sizes = []
            for m in modalities:
                start, end = config["ref_idx"][m]
                input_sizes.append(end - start)
                hidden_sizes.append(random.choice([32, 64, 128, 256]))

            model_config = {
                "input_sizes": input_sizes,
                "hidden_sizes": hidden_sizes,
                "batch_size": 32,
                "lr": random.choice([0.0001 ,0.0005, 0.001]),
                "dropout": random.choice([0.2, 0.5, 0.7]),
                "epochs": 150}
            models_config.append(model_config)


    best_model = None
    best_model_valdation_loss = 10e10
    for model_config in models_config:
        print("\nNew Model")
        print(model_config)
        input_sizes = model_config["input_sizes"]
        hidden_sizes = model_config["hidden_sizes"]
        lr = model_config["lr"]
        batch_size = model_config["batch_size"]
        dropout = model_config["dropout"]   
        epochs = model_config["epochs"]

        model = LFLSTM(modalities, input_sizes, hidden_sizes, output_size, dropout)
        optimizer = optim.Adam(model.parameters(),lr=lr)
        loss_fn = nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)
        model.to(device)

        best_validation_loss = 10e10
        num_trials = 3
        curr_patience = patience = 8

        for epoch in tqdm(range(epochs)):
            train_loss = train(model, X_train, y_train, loss_fn, batch_size)
            validation_loss = evaluate(model, X_valid, y_valid, loss_fn)
            scheduler.step(validation_loss)
            print(f"Epoch {epoch}: train loss {train_loss:.4f} | valid_loss: {validation_loss:.4f}") if epoch % 10 == 0 else None
        
            if validation_loss <= best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), os.path.join(save_path, "model_lf.std"))
                torch.save(optimizer.state_dict(), os.path.join(save_path, 'optim_lf.std'))
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    num_trials -= 1
                    curr_patience = patience
                    # ran out of patience, load previous best model
                    model.load_state_dict(torch.load(os.path.join(save_path, "model_lf.std")))
                    optimizer.load_state_dict(torch.load(os.path.join(save_path, "optim_lf.std")))
        
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break


        model.load_state_dict(torch.load(os.path.join(save_path, "model_lf.std")))
        print("Current Best Val Loss", best_validation_loss)

        if best_validation_loss < best_model_valdation_loss:
            best_model_valdation_loss = best_validation_loss
            best_model = model
    
    print("\n\nEvaluation Metric of Best Model\n" + "-" * 120)
    predictions, test_loss = predict(best_model, X_test, y_test, loss_fn)
    print("\n\nX_test")        
    evaluate_metrics(config["dataset_name"], predictions, y_test)
    print("-" * 120)
    if X_test_m != None:
        print("\n\nX_test_m")
        predictions_m, test_loss_m = predict(best_model, X_test_m, y_test, loss_fn)
        evaluate_metrics(config["dataset_name"], predictions_m, y_test)
    print("-" * 120)
    if X_test_s != None:
        predictions_s, test_loss_s = predict(best_model, X_test_s, y_test, loss_fn)
        print("\n\nX_test_s")
        evaluate_metrics(config["dataset_name"], predictions_s, y_test)
    
    print("-" * 120)
