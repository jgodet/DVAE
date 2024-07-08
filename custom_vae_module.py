# Credits to 2024 Luis Villamarin (luis.villamarin@etu.unistra.fr) For his work that was the basis of this VAE

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=4096, latent_dim=100, num_labels=10, use_labels_in_input = False, debug_mode=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.use_labels_in_input = use_labels_in_input
        self.debug_mode = debug_mode
        if use_labels_in_input : 
            self.full_input_dim = self.input_dim + self.num_labels
        else : 
            self.full_input_dim = self.input_dim

        super(ModularVAE, self).__init__()
        #encoder
        self.fc1 = nn.Linear(self.full_input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim) # mean of distribution
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim) # log of variance of distribution.
        #decoder
        self.fc3 = nn.Linear(latent_dim + num_labels, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # unused :
        self.fc5 = nn.Linear(latent_dim, num_labels)  

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # calculate standart deviation from logvar
        eps = torch.randn_like(std) # std is used to adjust the size of the tensor eps, but standart deviation of eps is 1
        if self.debug_mode : 
            self.nan_detector(eps, "eps")
            self.nan_detector(std, "std")
            self.nan_detector(mu + eps * std, "mu + eps * std")
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))   # remove sigmoid ?

    def predict_labels(self, z):
        return F.softmax(self.fc5(z), dim=1)

    def forward(self, x, y):
        if self.debug_mode :
            print(f"x.shape before view : {x.shape}")
        x = x.view(-1, self.input_dim)
        if self.debug_mode :
            print(f"x.shape after view : {x.shape}")
        y = F.one_hot(y, num_classes=self.num_labels).float()
        if self.use_labels_in_input :
            full_input = torch.cat((x, y), dim=1)
        else : 
            full_input = x

        h = F.relu(self.fc1(full_input))

        mu = self.fc2_mean(h)                  # those 2 lines are equivalent to encode method?
        logvar = self.fc2_logvar(h)            # mu, logvar = encode(h)

        z = self.reparameterize(mu, logvar)

        y_hat = self.predict_labels(z)

        if self.debug_mode : 
            print(f"z.shape : {z.shape}")
            print(f"y.shape : {y.shape}")

        # print("test1")
        combined_z_y = torch.cat((z, y), dim=1)
        # print("test2")
        
        recon_x = torch.sigmoid(self.fc4(F.relu(self.fc3(combined_z_y))))

        # temporary_tensor_1 = self.fc3(combined_z_y)
        # print("test2.5")
        # temporary_tensor = F.relu(temporary_tensor_1)
        # print("test3")
        # recon_x = torch.sigmoid(self.fc4(temporary_tensor))
        # print("test4")
        if self.debug_mode : 
            self.nan_detector(mu, "mu")
            self.nan_detector(logvar, "logvar")
            self.minmax_print(logvar, "logvar")
            self.nan_detector(combined_z_y, "combined_z_y")
            self.nan_detector(z, "z")
            self.nan_detector(recon_x, "recon_x")

        return recon_x, mu, logvar, y_hat   # this methods combine labels prediction and the reconstruction. consider splitting them.

    # combined loss function for reconstruction of x, and used to include y prediction from z. 
    # potential problem : changes on the encoding part can create losses through the y reconstruction part.
    def loss_function(self, recon_x, x, mu, logvar, beta=1):  
        if self.debug_mode :
            print(f"recon_x  : {recon_x.shape} , shape of x : {x.shape}")
            print("recon_x - Min:", torch.min(recon_x), "Max:", torch.max(recon_x), "Mean:", torch.mean(recon_x))
            print("x - Min:", torch.min(x), "Max:", torch.max(x), "Mean:", torch.mean(x))
            print("recon_x - Contains NaN:", torch.isnan(recon_x).any(), "Contains Inf:", torch.isinf(recon_x).any())
            print("x - Contains NaN:", torch.isnan(x).any(), "Contains Inf:", torch.isinf(x).any())
            print(f" recon_x  Min:", torch.min(recon_x), "Max:", torch.max(recon_x), "Mean:", torch.mean(recon_x))
            print(f" x  Min:", torch.min(x), "Max:", torch.max(x), "Mean:", torch.mean(x))
        MSE = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (beta * KLD + MSE).mean()
        correct = 42 #placeholder , to be removed.
        return loss, correct, KLD.item(), MSE.item()
    
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






def train_modular_vae(model, train_loader, optimizer, epoch, beta, device):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # Move data and label to the device
        data, label = data.to(device), label.to(device)

        # Check for NaNs and Infs in data and label
        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"Data contains NaNs or Infs at batch {batch_idx}")
        if torch.isnan(label).any() or torch.isinf(label).any():
            print(f"Labels contain NaNs or Infs at batch {batch_idx}")

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar, y_hat = model(data, label)

        # Check for NaNs and Infs in model outputs
        if torch.isnan(recon_batch).any() or torch.isinf(recon_batch).any():
            print(f"Recon_batch contains NaNs or Infs at batch {batch_idx}")
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"Mu contains NaNs or Infs at batch {batch_idx}")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print(f"Logvar contains NaNs or Infs at batch {batch_idx}")
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print(f"Y_hat contains NaNs or Infs at batch {batch_idx}")

        # Compute loss
        loss, correct_batch, kld, rec_loss = model.loss_function(recon_batch, data, mu, logvar, beta)

        # Check for NaNs and Infs in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Loss contains NaNs or Infs at batch {batch_idx}")

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += correct_batch

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset)}")

























