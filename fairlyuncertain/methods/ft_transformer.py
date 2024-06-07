import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from tab_transformer_pytorch import FTTransformer
import os

class FTTransformerModel(BaseEstimator):
    def __init__(self,
                 categories,
                 num_continuous,
                 dim = 32,                           # dimension, paper set at 32
                 dim_out = 2,                        # binary prediction, but could be anything
                 depth = 6,                          # depth, paper recommended 6
                 heads = 8,                          # heads, paper recommends 8
                 attn_dropout = 0.1,                 # post-attention dropout
                 ff_dropout = 0.1,                   # feed forward dropout
                 batch_size=1,
                 num_epochs=50,
                 lr=3e-4,
                 load_best_model_when_trained=True,
                 verbose=False,
                 loss_fn=None,
                 squeeze_output=False,
                 model_save_path='model_cache/best_model.pth'):  # Add model_save_path parameter
        super(FTTransformerModel, self).__init__()

        self.verbose = verbose
        self.dim = dim
        self.dim_out = dim_out
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.load_best_model_when_trained = load_best_model_when_trained
        self.best_model_dict = None
        self.loss_fn = loss_fn
        self.squeeze_output = squeeze_output
        self.model_save_path = model_save_path  # Initialize model save path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = self.build_model(
            categories,
            num_continuous,
            dim,
            dim_out,
            depth,
            heads,
            attn_dropout,
            ff_dropout,
        ).to(self.device)

    def build_model(self, categories, num_continuous, dim, dim_out, depth, heads, attn_dropout, ff_dropout):
        model = FTTransformer(
            categories = categories,
            num_continuous = num_continuous,
            dim = dim,
            dim_out = dim_out,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        return model

    def fit(self,
            X_cat_train,
            X_cont_train,
            y_train,
            patience=2,
            accumulation_steps=1):
        
        torch.autocast(device_type='cuda', dtype=torch.float16)

        X_cat_train, X_cont_train, y_train = X_cat_train.to(self.device), X_cont_train.to(self.device), y_train.to(self.device)
        
        X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = train_test_split(X_cat_train,
                                                                                            X_cont_train,
                                                                                            y_train,
                                                                                            test_size=0.1,
                                                                                            random_state=42)
        
        train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_cat_val, X_cont_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scaler = GradScaler()  # Initialize the gradient scaler for mixed precision

        best_accuracy = 0.0
        best_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            if self.verbose:
                batch_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
            else:
                batch_progress = train_loader

            optimizer.zero_grad()
            for i, (X_batch_cat, X_batch_cont, y_batch) in enumerate(batch_progress):
                X_batch_cat, X_batch_cont, y_batch = X_batch_cat.to(self.device), X_batch_cont.to(self.device), y_batch.to(self.device)

                with autocast(): 
                    output = self.model(X_batch_cat, X_batch_cont)
                    y_batch = y_batch.type(torch.LongTensor).to(self.device)
                    if self.squeeze_output:
                        output = output.squeeze()
                    loss = self.loss_fn(output, y_batch.float())

                scaler.scale(loss).backward() 

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer) 
                    scaler.update() 
                    optimizer.zero_grad()

                if self.verbose:
                    batch_progress.set_description(f'Epoch {epoch+1}/{self.num_epochs} Loss: {loss.item():.4f}')
                    batch_progress.refresh()

            if (i + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch_cat, X_batch_cont, y_batch in val_loader:
                    X_batch_cat, X_batch_cont, y_batch = X_batch_cat.to(self.device), X_batch_cont.to(self.device), y_batch.to(self.device)
                    with autocast():
                        val_output = self.model(X_batch_cat, X_batch_cont)
                        if self.squeeze_output:
                            val_output = val_output.squeeze()
                        y_batch = y_batch.type(torch.LongTensor).to(self.device)
                        batch_val_loss = self.loss_fn(val_output, y_batch.float()).item()
                        val_loss += batch_val_loss * y_batch.size(0)
                        if self.squeeze_output:
                            predict_proba = val_output
                        else:
                            predict_proba = val_output[:, 0]
                        val_preds_binary = (predict_proba > 0.5).cpu().numpy().astype(int)
                        val_correct += (val_preds_binary == y_batch.cpu().numpy()).sum()
                        val_total += y_batch.size(0)

            val_loss /= val_total
            val_accuracy = val_correct / val_total

            if val_loss < best_loss:
                best_loss = val_loss
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), self.model_save_path)
                if self.verbose:
                    print(f"Val loss - new best: {best_loss}")
                    print(f"Val accuracy - new best: {best_accuracy}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Epochs without improvement: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {patience} consecutive epochs.")
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Epochs without improvement: {epochs_without_improvement}/{patience}")
                print(f"Val accuracy - new best: {best_accuracy}")
                break

            with torch.no_grad():
                torch.cuda.empty_cache()

        if self.load_best_model_when_trained and os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))

    def load_best_model(self):
        """Loads the best model parameters."""
        if os.path.exists(self.model_save_path):
            self.model.load_state_dict(torch.load(self.model_save_path))
        else:
            print("No best model saved. Please run the training first.")

    def predict(self, X_cat_test, X_cont_test, binary=False, threshold=0.5):
        with torch.no_grad():
            X_cat_test, X_cont_test = X_cat_test.to(self.device), X_cont_test.to(self.device)
            predictions = self.model(X_cat_test, X_cont_test)
            if self.squeeze_output:
                predictions = predictions.squeeze()
            proba = predictions.softmax(dim=-1)
            if binary:
                predictions = (proba[:, 0] > threshold).cpu().numpy().astype(int)
            else:
                predictions = predictions.cpu().numpy()
            return predictions
        
    def predict_proba(self, X_cat_test, X_cont_test):
        with torch.no_grad():
            X_cat_test, X_cont_test = X_cat_test.to(self.device), X_cont_test.to(self.device)
            predictions = self.model(X_cat_test, X_cont_test)
            if self.squeeze_output:
                predictions = predictions.squeeze()
            proba = predictions.softmax(dim=-1)
            return proba.cpu().numpy()
