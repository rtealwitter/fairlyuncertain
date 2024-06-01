import torch
import numpy as np
import scipy
from torch.utils.data import DataLoader, Dataset

seed = 0
rng = np.random.default_rng(seed=seed)

# Neural Networks
class SelectiveSquare(torch.nn.Module): 
    def __init__(self): 
        super().__init__()
        return 

    def forward(self,x): 
        out = torch.zeros_like(x)
        out[:,0] = x[:,0]
        # Square the second variable (std is non-negative)
        out[:,1] = x[:,1]**2
        return out

class SeparateGradient(torch.nn.Module):
    def __init__(self, intermediate_size=100, num_layers=3, activation = torch.nn.ReLU):
        super().__init__()
        mean_layers = []
        std_layers = []
        for _ in range(num_layers):
            mean_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            mean_layers.append(activation())
            std_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            std_layers.append(activation())
        mean_layers.append(torch.nn.Linear(intermediate_size, 1))
        std_layers.append(torch.nn.Linear(intermediate_size, 1))
        self.mean_layers = torch.nn.Sequential(*mean_layers)
        self.std_layers = torch.nn.Sequential(*std_layers)
        return

    def forward(self,x):
        mean = self.mean_layers(x).squeeze()
        std = self.std_layers(x.detach()).squeeze()
        out = torch.zeros((x.shape[0], 2))
        out[:,0] = mean
        out[:,1] = std **2
        return out
    
def build_separate_model(input_size, num_layers, activation = torch.nn.ReLU, dropout=0, intermediate_size=100):
    layers = []
    layers.append(torch.nn.Linear(input_size, intermediate_size))
    layers.append(activation())
    for _ in range(num_layers//2):
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
        layers.append(activation())
    layers.append(SeparateGradient(intermediate_size, num_layers//2, activation))
    return torch.nn.Sequential(*layers)

def build_model(input_size, output_size, num_layers, activation = torch.nn.ReLU, dropout=0, intermediate_size=100):
    layers = []
    layers.append(torch.nn.Linear(input_size, intermediate_size))
    layers.append(activation())
    for _ in range(num_layers):
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
        layers.append(activation())
    layers.append(torch.nn.Linear(intermediate_size, output_size))
    if output_size == 2:
        layers.append(SelectiveSquare())
    else:
        # Logits
        layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers)
    model.train()
    return model

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(X, y, model, loss_fn, num_epochs, num_batches, learning_rate):
    n = X.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # Train the model
    batch_size = n // num_batches
    for t in range(num_epochs):
        # Shuffle the indices
        indices = rng.choice(n, n, replace=False)
        for batch in range(0, n, batch_size):
            # Compute the batch indices
            batch_indices = indices[batch:batch+batch_size]
            X_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]
            # Forward pass: compute predicted y
            pred = model(X_batch)
            loss = loss_fn(pred.squeeze(), y_batch)
            # Backward pass: zero, backpropagate and step
            model.zero_grad()
            loss.backward()
            optimizer.step()

    easy_input_model = lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    return easy_input_model

def normal_nll(pred, y_batch):
    mean_pred, std_pred = pred[:,0], pred[:,1]
    std_pred += 1e-6 # Avoid division by zero

    # Compute loss
    loss= (mean_pred - y_batch)**2 / (2*std_pred**2) + torch.log((std_pred)**2)

    return loss.mean()

def beta_nll(pred, y_batch):
    beta = .5
    mean_pred, std_pred = pred[:,0], pred[:,1]
    weighting = (std_pred**(2*beta)).detach()


    # Compute loss
    loss= weighting*((mean_pred - y_batch)**2 / (2*std_pred) + torch.log((std_pred)**2))

    return loss.mean()

def faithful_nll(pred, y_batch):
    mean_pred, std_pred = pred[:,0], pred[:,1]

    std_pred += 1e-6 # Avoid division by zero

    loss_mean = (mean_pred - y_batch)**2
    
    loss_std = (mean_pred.detach() - y_batch)**2 / (2*std_pred**2) + torch.log((std_pred)**2)

    # Compute loss
    loss = loss_mean + loss_std

    return loss.mean()

def binomial(pred, y_batch):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, y_batch)

def squared_error(pred, y_batch):
    return torch.nn.functional.mse_loss(pred, y_batch)

def nn_train(loss_name, nn_instance):
    if 'num_batches' not in nn_instance:
        nn_instance['num_batches'] = 5
    input_size = nn_instance['X_train'].shape[1]
    dropout = 0 if 'Dropout' not in loss_name else 0.2
    output_size = 1 if 'BCE' in loss_name else 2

    if 'Faithful' in loss_name:
        model = build_separate_model(input_size, nn_instance['num_layers'], dropout=dropout)
    else:
        model = build_model(input_size, output_size, nn_instance['num_layers'], dropout=dropout)

    model = train(nn_instance['X_train'], nn_instance['y_train'], model, nn_losses[loss_name], nn_instance['num_epochs'], nn_instance['num_batches'])

    return model

losses = {
    'Normal NLL' : normal_nll,
    'Faithful NLL' : faithful_nll,
    r'$\beta$-NLL' : beta_nll,
    'Logistic' : binomial,
    'Squared Error' : squared_error
}

class Model:
    def __init__(self, loss_name=None, instance=None):
        if instance is None: instance = {}

        # Model parameters
        depth = 6 if 'depth' not in instance else instance['depth']
        dropout = 0 if 'dropout' not in instance else instance['dropout']
        input_size = instance['X_train'].shape[1]
        output_size = 1 if loss_name in ['Logistic', 'Squared Error'] else 2

        if 'Faithful' in loss_name:
            self.model = build_separate_model(input_size, depth, dropout=dropout)
        else:
            self.model = build_model(input_size, output_size, depth, dropout=dropout)
        
        # Training parameters
        self.epochs = 100 if 'epochs' not in instance else instance['epochs']
        self.learning_rate = 0.01 if 'learning_rate' not in instance else instance['learning_rate']
        self.num_batches = 5 if 'num_batches' not in instance else instance['num_batches']
        self.loss = losses[loss_name]
        self.loss_name = loss_name
    
    def fit(self, X, y):
        self.model = train(X, y, self.model, self.loss, self.epochs, self.num_batches, self.learning_rate)
            
    def predict(self, X):
        if self.loss_name == 'Logistic':
            return self.model(X).round()
        return self.model(X)
 
    def predict_proba(self, X):
        return self.model(X)
