"""
Feature Extraction Module for Water Potability Prediction
Implements Autoencoder and TabTransformer for advanced feature learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Autoencoder(nn.Module):
    """
    Deep Autoencoder for feature extraction and dimensionality reduction
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 32, 
                 hidden_dims: list = [128, 64]):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers.extend([
            nn.Linear(prev_dim, latent_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Extract latent features"""
        return self.encoder(x)


class AutoencoderFeatureExtractor:
    """
    Autoencoder-based feature extraction
    """
    
    def __init__(self, latent_dim: int = 32, hidden_dims: list = [128, 64],
                 epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X: pd.DataFrame) -> 'AutoencoderFeatureExtractor':
        """Train autoencoder"""
        print("Training Autoencoder...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize autoencoder
        self.autoencoder = Autoencoder(
            input_dim=X.shape[1],
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.autoencoder(batch_x)
                loss = criterion(reconstructed, batch_x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        print("Autoencoder training completed!")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract latent features"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not fitted yet")
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Extract features
        self.autoencoder.eval()
        with torch.no_grad():
            latent_features = self.autoencoder.encode(X_tensor)
        
        # Convert to DataFrame
        latent_features = latent_features.cpu().numpy()
        feature_names = [f'autoencoder_{i}' for i in range(self.latent_dim)]
        
        return pd.DataFrame(latent_features, columns=feature_names)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for TabTransformer
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_linear(context)
        
        return output


class TabTransformer(nn.Module):
    """
    TabTransformer architecture for tabular data
    """
    
    def __init__(self, num_features: int, embed_dim: int = 64, 
                 num_heads: int = 8, num_layers: int = 3, 
                 dropout: float = 0.1):
        super(TabTransformer, self).__init__()
        
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Feature embedding layer
        self.feature_embedding = nn.Linear(1, embed_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_features, embed_dim)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim)
                ),
                'norm2': nn.LayerNorm(embed_dim),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for feature-wise processing
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
        
        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, num_features, embed_dim)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer layers
        for layer in self.transformer_layers:
            # Multi-head attention
            attn_output = layer['attention'](x)
            x = layer['norm1'](x + layer['dropout'](attn_output))
            
            # Feed-forward
            ff_output = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_output))
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        return x


class TabTransformerFeatureExtractor:
    """
    TabTransformer-based feature extraction
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1,
                 epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tabtransformer = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TabTransformerFeatureExtractor':
        """Train TabTransformer"""
        print("Training TabTransformer...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Create dataset and dataloader
        if y is not None:
            y_tensor = torch.FloatTensor(y.values).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize TabTransformer
        self.tabtransformer = TabTransformer(
            num_features=X.shape[1],
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.tabtransformer.parameters(), lr=self.learning_rate)
        
        # Self-supervised training (reconstruction)
        criterion = nn.MSELoss()
        
        # Training loop
        self.tabtransformer.train()
        for epoch in range(self.epochs):
            total_loss = 0
            
            for batch_data in dataloader:
                if len(batch_data) == 2:
                    batch_x, _ = batch_data
                else:
                    batch_x, = batch_data
                
                optimizer.zero_grad()
                
                # Forward pass (self-reconstruction)
                features = self.tabtransformer(batch_x)
                
                # Simple reconstruction loss (project back to input space)
                reconstructed = torch.matmul(features, self.tabtransformer.feature_embedding.weight.squeeze())
                loss = criterion(reconstructed, batch_x)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        print("TabTransformer training completed!")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features using TabTransformer"""
        if self.tabtransformer is None:
            raise ValueError("TabTransformer not fitted yet")
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Extract features
        self.tabtransformer.eval()
        with torch.no_grad():
            features = self.tabtransformer(X_tensor)
        
        # Convert to DataFrame
        features = features.cpu().numpy()
        feature_names = [f'tabtransformer_{i}' for i in range(self.embed_dim)]
        
        return pd.DataFrame(features, columns=feature_names)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class EnsembleFeatureExtractor:
    """
    Ensemble of Autoencoder and TabTransformer feature extractors
    """
    
    def __init__(self, autoencoder_params: Optional[dict] = None,
                 tabtransformer_params: Optional[dict] = None,
                 weights: list = [0.5, 0.5]):
        self.autoencoder_params = autoencoder_params or {}
        self.tabtransformer_params = tabtransformer_params or {}
        self.weights = weights
        
        self.autoencoder_extractor = AutoencoderFeatureExtractor(**self.autoencoder_params)
        self.tabtransformer_extractor = TabTransformerFeatureExtractor(**self.tabtransformer_params)
        
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EnsembleFeatureExtractor':
        """Fit both extractors"""
        print("Fitting Ensemble Feature Extractor...")
        
        # Fit autoencoder
        self.autoencoder_extractor.fit(X)
        
        # Fit TabTransformer
        self.tabtransformer_extractor.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract ensemble features"""
        if not self.is_fitted:
            raise ValueError("EnsembleFeatureExtractor not fitted yet")
        
        # Extract features from both models
        autoencoder_features = self.autoencoder_extractor.transform(X)
        tabtransformer_features = self.tabtransformer_extractor.transform(X)
        
        # Concatenate features
        ensemble_features = pd.concat([
            autoencoder_features * self.weights[0],
            tabtransformer_features * self.weights[1]
        ], axis=1)
        
        return ensemble_features
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    print("Original data shape:", X.shape)
    
    # Test Autoencoder feature extraction
    print("\n" + "="*50)
    print("Testing Autoencoder Feature Extraction")
    print("="*50)
    
    ae_extractor = AutoencoderFeatureExtractor(latent_dim=16, epochs=50)
    ae_features = ae_extractor.fit_transform(X)
    print("Autoencoder features shape:", ae_features.shape)
    
    # Test TabTransformer feature extraction
    print("\n" + "="*50)
    print("Testing TabTransformer Feature Extraction")
    print("="*50)
    
    tt_extractor = TabTransformerFeatureExtractor(embed_dim=32, epochs=30)
    tt_features = tt_extractor.fit_transform(X, y)
    print("TabTransformer features shape:", tt_features.shape)
    
    # Test Ensemble feature extraction
    print("\n" + "="*50)
    print("Testing Ensemble Feature Extraction")
    print("="*50)
    
    ensemble_extractor = EnsembleFeatureExtractor(
        autoencoder_params={'latent_dim': 16, 'epochs': 30},
        tabtransformer_params={'embed_dim': 32, 'epochs': 20},
        weights=[0.4, 0.6]
    )
    ensemble_features = ensemble_extractor.fit_transform(X, y)
    print("Ensemble features shape:", ensemble_features.shape)
