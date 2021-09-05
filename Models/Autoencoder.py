import torch
import torch.nn as nn
import os
from tqdm.notebook import tqdm
import random
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

save_path = os.getcwd()
save_path = os.path.join(save_path, "Models/SavedModels")

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):

    def __init__(self, target_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.target_size = target_size
        self.hidden_size = hidden_size
        
        self.lstmcell = nn.LSTMCell(target_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, target_size)

    def forward(self, x, hidden, cell):
        hidden = hidden.squeeze()
        cell = cell.squeeze()
        # input size is going to be B x D i.e X_batch[:, t, :]
        hidden, cell = self.lstmcell(x, (hidden, cell))
        x = self.fc1(hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x, hidden, cell

        

class LSTM_AE(nn.Module):

    def __init__(self, encoder, decoder, enforce_rate):
        super(LSTM_AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enforce_rate = enforce_rate

    def forward(self, source, target):
        batch_size = source.shape[0]
        T = source.shape[1]

        output = torch.zeros_like(target).cuda()

        hidden, cell = self.encoder(source)

        x = torch.zeros_like(target[:, 0, :])

        for t in range(T):
            x, hidden, cell = self.decoder(x, hidden, cell)
            output[:, t, :] = x
            if self.training:
                x = target[:, t, :] if random.random() < self.enforce_rate else x

        return output

def train_ae(exp_config, src_to_tgt, model_config = None):

    def train(model, src_train, tgt_train, loss_fn):
        epoch_loss = 0
        model.train()
        N = src_train.shape[0]
        num_batches = 0
        loader = DataLoader(range(N), batch_size, shuffle = False)
        for idx in loader:
            num_batches += 1
            optimizer.zero_grad()
            src_batch = src_train[idx].to(device)
            tgt_batch = tgt_train[idx].to(device)
            tgt_hat = model(src_batch, tgt_batch)
            loss = loss_fn(tgt_hat, tgt_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches
        
    def evaluate(model, src_valid, tgt_valid, loss_fn):
        epoch_loss = 0
        model.eval()
        N = src_valid.shape[0]
        with torch.no_grad():
            src_batch = src_valid.to(device); tgt_batch = tgt_valid.to(device)
            tgt_hat = model(src_batch, tgt_batch)
            loss = loss_fn(tgt_hat, tgt_batch)
            epoch_loss += loss.item()
    
        return epoch_loss

    src_modality, tgt_modality = src_to_tgt.split("->")

    src_train = exp_config["src_train"][src_modality]
    tgt_train = exp_config["tgt_train"][tgt_modality]

    src_valid = exp_config["src_valid"][src_modality]
    tgt_valid = exp_config["tgt_valid"][tgt_modality]

    src_size = src_valid.shape[2]
    tgt_size = tgt_valid.shape[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    models_config = []

    if model_config:
        models_config.append(model_config)
    else:
        for i in range(5):
            #randomly create 3 configurations to train the model
            model_config = {
                "batch_size": 32,
                "lr": random.choice([0.0001, 0.0005, 0.001]),
                "hidden_size": random.choice([64, 128, 256]),
                "dropout": random.choice([0.2, 0.5, 0.8]),
                "epochs": 100,
                "enforce_rate": random.choice([0.2, 0.5, 0.7])
                }
            models_config.append(model_config)
        
    best_model_validation_loss = 10e10
    best_model = None
    print("Currently training for", f"{src_modality}->{tgt_modality}")
    print("=" * 120)
    for model_config in models_config:
        print("\nNew Model")
        # get current model hyper parameters
        print(model_config)
        epochs = model_config["epochs"]
        batch_size = model_config["batch_size"]
        hidden_size = model_config["hidden_size"]
        dropout = model_config["dropout"]
        enforce_rate = model_config["enforce_rate"]
        lr = model_config["lr"]

        # build model
        encoder = Encoder(src_size, hidden_size).to(device)
        decoder = Decoder(tgt_size, hidden_size, dropout).to(device)
        model = LSTM_AE(encoder, decoder, enforce_rate).to(device)
        optimizer = optim.Adam(model.parameters(),lr=lr)
        loss_fn = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

        # training
        best_validation_loss = 10e10
        num_trials = 2
        curr_patience = patience = 5
        for epoch in tqdm(range(1, epochs+1)):
            train_loss = train(model, src_train, tgt_train, loss_fn)
            validation_loss = evaluate(model, src_valid, tgt_valid, loss_fn)
            scheduler.step(validation_loss)
            print(f"Epoch {epoch}: train loss {train_loss:.4f} | valid_loss: {validation_loss:.4f}") if epoch % 10 == 0 else None

            if validation_loss <= best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), os.path.join(save_path, "model_ae.std"))
                torch.save(optimizer.state_dict(), os.path.join(save_path, 'optim_ae.std'))
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    num_trials -= 1
                    curr_patience = patience
                    model.load_state_dict(torch.load(os.path.join(save_path, "model_ae.std")))
                    optimizer.load_state_dict(torch.load(os.path.join(save_path, "optim_ae.std")))
        
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break
        # load current model best performed iteration
        model.load_state_dict(torch.load(os.path.join(save_path, "model_ae.std")))
        # update best model
        print("Current Best Val Loss", best_validation_loss)
        if best_validation_loss < best_model_validation_loss:
            best_model_validation_loss = best_validation_loss
            model.load_state_dict(torch.load(os.path.join(save_path, "model_ae.std")))
            best_model = model
    print("=" * 120)

    exp_config["Autoencoder"][src_to_tgt] = best_model
