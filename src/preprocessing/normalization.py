"""
Advanced Normalization Module for Water Potability Prediction
Implements Quantile Transformer, Copula Normalization, and Robust Scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from scipy import stats
from scipy.stats import norm, gaussian_kde
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class QuantileNormalizer:
    """
    Quantile Transformer for non-parametric normalization
    """
    
    def __init__(self, output_distribution: str = 'uniform', 
                 n_quantiles: int = 1000):
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.quantile_transformer = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame) -> 'QuantileNormalizer':
        """Fit quantile transformer"""
        self.feature_names = X.columns.tolist()
        self.quantile_transformer = QuantileTransformer(
            output_distribution=self.output_distribution,
            n_quantiles=min(self.n_quantiles, len(X)),
            random_state=42
        )
        self.quantile_transformer.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using quantile transformer"""
        if self.quantile_transformer is None:
            raise ValueError("QuantileNormalizer not fitted yet")
        
        X_transformed = self.quantile_transformer.transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform to original scale"""
        if self.quantile_transformer is None:
            raise ValueError("QuantileNormalizer not fitted yet")
        
        X_original = self.quantile_transformer.inverse_transform(X)
        return pd.DataFrame(X_original, columns=self.feature_names)


class CopulaNormalizer:
    """
    Copula-based normalization preserving dependency structure
    """
    
    def __init__(self, copula_type: str = 'gaussian'):
        self.copula_type = copula_type
        self.marginal_transforms = {}
        self.copula_params = None
        self.feature_names = None
        
    def _fit_marginal(self, data: np.ndarray) -> Tuple:
        """Fit marginal distribution"""
        # Use empirical CDF for marginal transformation
        sorted_data = np.sort(data)
        n = len(data)
        
        # Create empirical CDF
        def empirical_cdf(x):
            return np.searchsorted(sorted_data, x, side='right') / n
        
        # Create inverse CDF
        def inverse_cdf(u):
            # Ensure u is in valid range
            u = np.clip(u, 1e-10, 1 - 1e-10)
            indices = (u * (n - 1)).astype(int)
            return sorted_data[indices]
        
        return empirical_cdf, inverse_cdf
    
    def fit(self, X: pd.DataFrame) -> 'CopulaNormalizer':
        """Fit copula normalizer"""
        self.feature_names = X.columns.tolist()
        X_np = X.values
        
        # Fit marginal distributions
        for i, col in enumerate(self.feature_names):
            empirical_cdf, inverse_cdf = self._fit_marginal(X_np[:, i])
            self.marginal_transforms[col] = {
                'cdf': empirical_cdf,
                'inverse_cdf': inverse_cdf
            }
        
        # Transform to uniform marginals
        X_uniform = np.zeros_like(X_np)
        for i, col in enumerate(self.feature_names):
            X_uniform[:, i] = np.array([self.marginal_transforms[col]['cdf'](x) 
                                       for x in X_np[:, i]])
        
        # Fit copula parameters (Gaussian copula)
        if self.copula_type == 'gaussian':
            # Transform to normal using inverse normal CDF
            X_normal = norm.ppf(np.clip(X_uniform, 1e-10, 1 - 1e-10))
            
            # Estimate correlation matrix
            self.copula_params = np.corrcoef(X_normal.T)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using copula normalization"""
        if self.copula_params is None:
            raise ValueError("CopulaNormalizer not fitted yet")
        
        X_np = X.values
        X_transformed = np.zeros_like(X_np)
        
        # Transform to uniform marginals
        for i, col in enumerate(self.feature_names):
            X_transformed[:, i] = np.array([self.marginal_transforms[col]['cdf'](x) 
                                          for x in X_np[:, i]])
        
        # Transform to normal using copula
        if self.copula_type == 'gaussian':
            X_transformed = norm.ppf(np.clip(X_transformed, 1e-10, 1 - 1e-10))
        
        return pd.DataFrame(X_transformed, columns=self.feature_names)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class RobustNormalizer:
    """
    Robust scaling using median and IQR
    """
    
    def __init__(self, with_centering: bool = True, with_scaling: bool = True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.robust_scaler = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame) -> 'RobustNormalizer':
        """Fit robust scaler"""
        self.feature_names = X.columns.tolist()
        self.robust_scaler = RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling
        )
        self.robust_scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using robust scaling"""
        if self.robust_scaler is None:
            raise ValueError("RobustNormalizer not fitted yet")
        
        X_transformed = self.robust_scaler.transform(X)
        return pd.DataFrame(X_transformed, columns=self.feature_names)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class EnsembleNormalizer:
    """
    Ensemble of multiple normalization techniques
    """
    
    def __init__(self, methods: list = ['quantile', 'copula', 'robust'],
                 weights: Optional[list] = None):
        self.methods = methods
        self.weights = weights or [1.0 / len(methods)] * len(methods)
        self.normalizers = {}
        self.feature_names = None
        
        # Initialize normalizers
        if 'quantile' in methods:
            self.normalizers['quantile'] = QuantileNormalizer()
        if 'copula' in methods:
            self.normalizers['copula'] = CopulaNormalizer()
        if 'robust' in methods:
            self.normalizers['robust'] = RobustNormalizer()
    
    def fit(self, X: pd.DataFrame) -> 'EnsembleNormalizer':
        """Fit all normalizers"""
        self.feature_names = X.columns.tolist()
        
        for method_name, normalizer in self.normalizers.items():
            print(f"Fitting {method_name} normalizer...")
            normalizer.fit(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using ensemble of normalizers"""
        if not self.feature_names:
            raise ValueError("EnsembleNormalizer not fitted yet")
        
        transformed_data = []
        
        for method_name, normalizer in self.normalizers.items():
            print(f"Applying {method_name} transformation...")
            X_transformed = normalizer.transform(X)
            transformed_data.append(X_transformed.values)
        
        # Weighted ensemble
        transformed_data = np.array(transformed_data)
        weights = np.array(self.weights).reshape(-1, 1, 1)
        
        ensemble_result = np.sum(transformed_data * weights, axis=0)
        
        return pd.DataFrame(ensemble_result, columns=self.feature_names)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def get_normalization_report(self) -> dict:
        """Get normalization report"""
        report = {
            'methods_used': list(self.normalizers.keys()),
            'weights': self.weights,
            'num_features': len(self.feature_names) if self.feature_names else 0
        }
        
        return report


class AdaptiveNormalizer:
    """
    Adaptive normalization that selects best method based on data characteristics
    """
    
    def __init__(self):
        self.best_normalizer = None
        self.best_method = None
        self.feature_names = None
        self.normalizers = {
            'quantile': QuantileNormalizer(),
            'robust': RobustNormalizer(),
            'copula': CopulaNormalizer()
        }
    
    def _evaluate_normalization(self, X_normalized: pd.DataFrame) -> float:
        """Evaluate normalization quality"""
        # Check for normality using Shapiro-Wilk test
        normality_scores = []
        
        for col in X_normalized.columns:
            if len(X_normalized[col]) > 5000:  # Sample for large datasets
                sample = X_normalized[col].sample(5000, random_state=42)
            else:
                sample = X_normalized[col]
            
            try:
                stat, p_value = stats.shapiro(sample)
                normality_scores.append(p_value)
            except:
                normality_scores.append(0.0)
        
        return np.mean(normality_scores)
    
    def fit(self, X: pd.DataFrame) -> 'AdaptiveNormalizer':
        """Fit and select best normalizer"""
        self.feature_names = X.columns.tolist()
        best_score = -np.inf
        
        for method_name, normalizer in self.normalizers.items():
            try:
                print(f"Evaluating {method_name} normalization...")
                X_normalized = normalizer.fit_transform(X)
                score = self._evaluate_normalization(X_normalized)
                
                print(f"{method_name} normalization score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    self.best_normalizer = normalizer
                    self.best_method = method_name
                    
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                continue
        
        print(f"Selected best normalization method: {self.best_method}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using best normalizer"""
        if self.best_normalizer is None:
            raise ValueError("AdaptiveNormalizer not fitted yet")
        
        return self.best_normalizer.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class NormalizationPipeline:
    """
    Complete normalization pipeline with multiple techniques
    """
    
    def __init__(self, method: str = 'ensemble', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.normalizer = None
        self.is_fitted = False
        
        # Initialize normalizer based on method
        if method == 'quantile':
            self.normalizer = QuantileNormalizer(**kwargs)
        elif method == 'copula':
            self.normalizer = CopulaNormalizer(**kwargs)
        elif method == 'robust':
            self.normalizer = RobustNormalizer(**kwargs)
        elif method == 'ensemble':
            self.normalizer = EnsembleNormalizer(**kwargs)
        elif method == 'adaptive':
            self.normalizer = AdaptiveNormalizer()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit(self, X: pd.DataFrame) -> 'NormalizationPipeline':
        """Fit normalization pipeline"""
        print(f"Fitting {self.method} normalization pipeline...")
        self.normalizer.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        if not self.is_fitted:
            raise ValueError("NormalizationPipeline not fitted yet")
        
        return self.normalizer.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def get_normalization_info(self) -> dict:
        """Get normalization information"""
        info = {
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        
        if hasattr(self.normalizer, 'get_normalization_report'):
            info.update(self.normalizer.get_normalization_report())
        elif hasattr(self.normalizer, 'best_method'):
            info['best_method'] = self.normalizer.best_method
        
        return info


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create sample data with different distributions
    np.random.seed(42)
    n_samples = 1000
    
    # Create mixed distribution data
    data = {
        'normal_feature': np.random.normal(0, 1, n_samples),
        'uniform_feature': np.random.uniform(-2, 2, n_samples),
        'exponential_feature': np.random.exponential(1, n_samples),
        'skewed_feature': np.random.gamma(2, 2, n_samples)
    }
    
    X = pd.DataFrame(data)
    
    print("Original data statistics:")
    print(X.describe())
    
    # Test different normalization methods
    methods = ['quantile', 'robust', 'ensemble', 'adaptive']
    
    for method in methods:
        print(f"\n{'='*40}")
        print(f"Testing {method} normalization:")
        print(f"{'='*40}")
        
        normalizer = NormalizationPipeline(method=method)
        X_normalized = normalizer.fit_transform(X)
        
        print("Normalized data statistics:")
        print(X_normalized.describe())
        
        # Get normalization info
        info = normalizer.get_normalization_info()
        print(f"Normalization info: {info}")
