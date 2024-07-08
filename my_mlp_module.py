# Credits to 2024 Luis Villamarin (luis.villamarin@etu.unistra.fr) For his work that was the basis of this VAE

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np


class MyMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=None,  num_labels=10,  debug_mode=False):

        if hidden_dim == None :
            self.hidden_dim = (input_dim + num_labels) // 2
        else :
            self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.debug_mode = debug_mode

        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_labels) # mean of distribution

    def predict_labels(self, x):
        h = F.relu(self.fc1(x))
        return F.softmax(self.fc2(h), dim=1)

    def forward(self, x):
        if self.debug_mode :
            print(f"x.shape before view : {x.shape}")
        x = x.view(-1, self.input_dim)
        if self.debug_mode :
            print(f"x.shape after view : {x.shape}")
        h = F.relu(self.fc1(x))
        y_hat =  self.fc2(h)
        return y_hat   #  labels prediction

    def loss_function(self ,y_hat, y):  
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)

        return loss 

    def train_myself(self, train_loader, optimizer, epoch, device):
        self.train()
        train_loss = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            # Move data and label to the device
            data, label = data.to(device), label.to(device)

            y = F.one_hot(label, self.num_labels).float()

            # Check for NaNs and Infs in data and label
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Data contains NaNs or Infs at batch {batch_idx}")
            if torch.isnan(label).any() or torch.isinf(label).any():
                print(f"Labels contain NaNs or Infs at batch {batch_idx}")

            # Forward pass
            y_hat = self.forward(data)

            # Check for NaNs and Infs in model outputs
            if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                print(f"Y_hat contains NaNs or Infs at batch {batch_idx}")
            # Compute loss
            loss = self.loss_function(y_hat, y)

            # Check for NaNs and Infs in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Loss contains NaNs or Infs at batch {batch_idx}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}")


    def minmax_print(self, x , name = "this tensor"):
        print(f" {name}  Min:", torch.min(x), "Max:", torch.max(x), "Mean:", torch.mean(x))

    def nan_detector(self, x , name = "this tensor", verbose = True) :
        have_nan = torch.isnan(x).any()
        have_inf = torch.isinf(x).any()
        if have_nan :
            print(f"{name} Contains NaN ")
        if have_inf :
            print(f"{name} Contains Inf")
        if verbose and not(have_nan or have_inf) : 
            print(f"{name} have been checked for inf & nan")

    def evaluation(self, test_loader, device) : 
        all_labels = []
        all_preds = []
        self.eval()
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                y_hat = self(data)
                _, predicted = torch.max(y_hat.data, 1)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Accuracy: {accuracy:.4f}')

        # Precision, Recall, F1-score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        # Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        print('Confusion Matrix:')
        print(conf_matrix)

        # Classification Report
        class_report = classification_report(all_labels, all_preds)
        print('Classification Report:')
        print(class_report)

        print("returning : precision, recall, f1, conf_matrix, class_report")

        return precision, recall, f1, conf_matrix, class_report








class MyMLPFromLatentSpace(nn.Module):
    def __init__(self,  encoder, input_dim=784, hidden_dim=None,  num_labels=10, z_dim = 15, debug_mode=False):
        super(MyMLPFromLatentSpace, self).__init__()
        if hidden_dim == None :
            self.hidden_dim = (z_dim + num_labels) // 2
        else :
            self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.z_dim = z_dim
        self.encoder = encoder
        self.debug_mode = debug_mode


        self.fc1 = nn.Linear(self.z_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_labels) # mean of distribution

    def predict_labels(self, x):
        if self.debug_mode :
            print(f"is x a tensor ? {torch.is_tensor(x)}")
            print(f"type of x : {type(x)}")
        mu, logvar = self.encoder.encode(x)
        z = self.encoder.reparameterize(mu, logvar )
        if self.debug_mode :
            print(f"is z a tensor ? {torch.is_tensor(z)}")
            print(f"type of z : {type(z)}")
        h = F.relu(self.fc1(z))
        return F.softmax(self.fc2(h), dim=1)

    def forward(self, x):
        # z = encoder.encode(x)
        ############################### previous code : 
        # if self.debug_mode :
        #     print(f"x.shape before view : {x.shape}")
        # x = x.view(-1, self.input_dim)
        # if self.debug_mode :
        #     print(f"x.shape after view : {x.shape}")
        # h = F.relu(self.fc1(x))
        x = x.view(-1, self.input_dim)
        if self.debug_mode : 
            print(f"in forward : shape of x : {x.shape}")
        y_hat =  self.predict_labels(x)
        return y_hat   #  labels prediction

    def loss_function(self ,y_hat, y):  
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss 

    def train_myself(self, train_loader, optimizer, epoch, device):
        self.train()
        train_loss = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            # Move data and label to the device
            data, label = data.to(device), label.to(device)

            y = F.one_hot(label, self.num_labels).float()

            # Check for NaNs and Infs in data and label
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Data contains NaNs or Infs at batch {batch_idx}")
            if torch.isnan(label).any() or torch.isinf(label).any():
                print(f"Labels contain NaNs or Infs at batch {batch_idx}")

            # Forward pass
            y_hat = self.forward(data)

            # Check for NaNs and Infs in model outputs
            if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
                print(f"Y_hat contains NaNs or Infs at batch {batch_idx}")
            # Compute loss
            loss = self.loss_function(y_hat, y)

            # Check for NaNs and Infs in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Loss contains NaNs or Infs at batch {batch_idx}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}")


    def minmax_print(self, x , name = "this tensor"):
        print(f" {name}  Min:", torch.min(x), "Max:", torch.max(x), "Mean:", torch.mean(x))

    def nan_detector(self, x , name = "this tensor", verbose = True) :
        have_nan = torch.isnan(x).any()
        have_inf = torch.isinf(x).any()
        if have_nan :
            print(f"{name} Contains NaN ")
        if have_inf :
            print(f"{name} Contains Inf")
        if verbose and not(have_nan or have_inf) : 
            print(f"{name} have been checked for inf & nan")

    def evaluation(self, test_loader, device) : 
        all_labels = []
        all_preds = []
        self.eval()
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                y_hat = self(data)
                _, predicted = torch.max(y_hat.data, 1)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Accuracy: {accuracy:.4f}')

        # Precision, Recall, F1-score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        # Confusion Matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        print('Confusion Matrix:')
        print(conf_matrix)

        # Classification Report
        class_report = classification_report(all_labels, all_preds)
        print('Classification Report:')
        print(class_report)

        print("returning : precision, recall, f1, conf_matrix, class_report")

        return precision, recall, f1, conf_matrix, class_report
















