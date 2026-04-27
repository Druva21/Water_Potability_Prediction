"""
Data Cleaning Module for Water Potability Prediction
Implements GAIN imputation and Isolation Forest for outlier detection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GAINImputer:
    """
    Generative Adversarial Imputation Networks (GAIN) for missing data imputation
    """
    
    def __init__(self, data_dim: int, hidden_dim: int = 128, 
                 alpha: float = 100.0, epochs: int = 1000):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.epochs = epochs
        self.generator = None
        self.discriminator = None
        
    def build_generator(self) -> nn.Module:
        """Build generator network"""
        return nn.Sequential(
            nn.Linear(self.data_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.data_dim),
            nn.Sigmoid()
        )
    
    def build_discriminator(self) -> nn.Module:
        """Build discriminator network"""
        return nn.Sequential(
            nn.Linear(self.data_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.data_dim),
            nn.Sigmoid()
        )
    
    def impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform GAIN imputation"""
        # Convert to numpy and normalize
        X_np = X.values
        X_min = X_np.min(axis=0)
        X_max = X_np.max(axis=0)
        X_norm = (X_np - X_min) / (X_max - X_min + 1e-8)
        
        # Create mask (1 for observed, 0 for missing)
        mask = ~np.isnan(X_norm)
        mask = mask.astype(float)
        
        # Initialize missing values with random noise
        X_norm[np.isnan(X_norm)] = np.random.random((np.isnan(X_norm)).sum())
        
        # Build networks
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Optimizers
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_norm)
        mask_tensor = torch.FloatTensor(mask)
        
        for epoch in range(self.epochs):
            # Generate imputed data
            z = torch.randn_like(X_tensor)
            x_hat = self.generator(torch.cat([X_tensor, z], dim=1))
            
            # Discriminator step
            d_optimizer.zero_grad()
            d_real = self.discriminator(torch.cat([X_tensor, mask_tensor], dim=1))
            d_fake = self.discriminator(torch.cat([x_hat.detach(), mask_tensor], dim=1))
            d_loss = criterion(d_real, mask_tensor) + criterion(d_fake, 1 - mask_tensor)
            d_loss.backward()
            d_optimizer.step()
            
            # Generator step
            g_optimizer.zero_grad()
            g_loss = criterion(self.discriminator(torch.cat([x_hat, mask_tensor], dim=1)), 
                             mask_tensor) + self.alpha * torch.mean((x_hat - X_tensor) ** 2 * mask_tensor)
            g_loss.backward()
            g_optimizer.step()
            
            if epoch % 100 == 0:
                print(f"GAIN Epoch {epoch}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
        
        # Final imputation
        with torch.no_grad():
            z = torch.randn_like(X_tensor)
            x_imputed = self.generator(torch.cat([X_tensor, z], dim=1))
        
        # Convert back to original scale
        x_imputed = x_imputed.numpy()
        x_imputed = x_imputed * (X_max - X_min + 1e-8) + X_min
        
        # Keep original observed values
        result = X_np.copy()
        result[np.isnan(X_np)] = x_imputed[np.isnan(X_np)]
        
        return pd.DataFrame(result, columns=X.columns)


class OutlierDetector:
    """
    Outlier detection using Isolation Forest
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit outlier detector and remove outliers"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        outlier_labels = self.isolation_forest.fit_predict(X_scaled)
        
        # Keep only inliers (label = 1)
        inlier_mask = outlier_labels == 1
        X_clean = X[inlier_mask]
        
        print(f"Removed {(~inlier_mask).sum()} outliers out of {len(X)} samples")
        print(f"Remaining samples: {len(X_clean)}")
        
        return X_clean
    
    def get_outlier_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get outlier scores for samples"""
        if self.isolation_forest is None:
            raise ValueError("OutlierDetector not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return -self.isolation_forest.score_samples(X_scaled)


class DataCleaner:
    """
    Complete data cleaning pipeline
    """
    
    def __init__(self, gain_params: Optional[dict] = None, 
                 outlier_params: Optional[dict] = None):
        self.gain_params = gain_params or {}
        self.outlier_params = outlier_params or {}
        self.gain_imputer = None
        self.outlier_detector = None
        
    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Complete data cleaning pipeline"""
        print("Starting data cleaning...")
        print(f"Original data shape: {X.shape}")
        
        # Step 1: GAIN Imputation
        print("\n1. Performing GAIN imputation...")
        self.gain_imputer = GAINImputer(
            data_dim=X.shape[1],
            **self.gain_params
        )
        X_imputed = self.gain_imputer.impute(X)
        print(f"Data shape after imputation: {X_imputed.shape}")
        
        # Step 2: Outlier Detection
        print("\n2. Detecting and removing outliers...")
        self.outlier_detector = OutlierDetector(**self.outlier_params)
        X_clean = self.outlier_detector.fit_transform(X_imputed)
        print(f"Final data shape: {X_clean.shape}")
        
        return X_clean
    
    def get_data_quality_report(self, X_original: pd.DataFrame, 
                              X_clean: pd.DataFrame) -> dict:
        """Generate data quality report"""
        report = {
            'original_shape': X_original.shape,
            'clean_shape': X_clean.shape,
            'samples_removed': X_original.shape[0] - X_clean.shape[0],
            'removal_percentage': (1 - X_clean.shape[0] / X_original.shape[0]) * 100,
            'original_missing_values': X_original.isnull().sum().sum(),
            'final_missing_values': X_clean.isnull().sum().sum()
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create sample data with missing values
    X, _ = make_classification(n_samples=1000, n_features=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    # Add missing values
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan
    
    print("Sample data with missing values:")
    print(X.head())
    print(f"Missing values: {X.isnull().sum().sum()}")
    
    # Clean data
    cleaner = DataCleaner()
    X_clean = cleaner.clean_data(X)
    
    # Quality report
    report = cleaner.get_data_quality_report(X, X_clean)
    print("\nData Quality Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
