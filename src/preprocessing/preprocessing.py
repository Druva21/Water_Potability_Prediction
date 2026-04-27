"""
Advanced Preprocessing Module for Water Potability Prediction
Implements Robust PCA and Feature Engineering
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scipy import stats
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RobustPCATransformer:
    """
    Robust Principal Component Analysis for noise reduction
    """
    
    def __init__(self, n_components: float = 0.95, 
                 robust_threshold: float = 2.0):
        self.n_components = n_components
        self.robust_threshold = robust_threshold
        self.pca = None
        self.scaler = RobustScaler()
        self.component_thresholds = None
        
    def fit(self, X: pd.DataFrame) -> 'RobustPCATransformer':
        """Fit robust PCA"""
        # Scale data using robust scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        if self.n_components < 1.0:
            self.pca = PCA(n_components=self.n_components)
        else:
            self.pca = PCA(n_components=int(self.n_components))
            
        self.pca.fit(X_scaled)
        
        # Calculate robust thresholds for each component
        X_transformed = self.pca.transform(X_scaled)
        self.component_thresholds = np.percentile(
            np.abs(X_transformed), 
            95, 
            axis=0
        ) * self.robust_threshold
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using robust PCA"""
        if self.pca is None:
            raise ValueError("RobustPCATransformer not fitted yet")
        
        # Scale and transform
        X_scaled = self.scaler.transform(X)
        X_transformed = self.pca.transform(X_scaled)
        
        # Apply robust thresholding
        robust_mask = np.abs(X_transformed) <= self.component_thresholds
        X_robust = X_transformed * robust_mask
        
        # Convert back to DataFrame
        columns = [f'RPCA_{i}' for i in range(X_robust.shape[1])]
        return pd.DataFrame(X_robust, columns=columns)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)


class FeatureEngineer:
    """
    Advanced feature engineering for water potability prediction
    """
    
    def __init__(self):
        self.feature_names = None
        self.engineered_features = []
        
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features based on domain knowledge"""
        X_engineered = X.copy()
        
        # Water chemistry interactions
        if all(col in X.columns for col in ['ph', 'hardness']):
            X_engineered['ph_hardness_ratio'] = X['ph'] / (X['hardness'] + 1e-8)
        
        if all(col in X.columns for col in ['sulfate', 'chloramines']):
            X_engineered['sulfate_chloramines_product'] = X['sulfate'] * X['chloramines']
        
        if all(col in X.columns for col in ['solids', 'conductivity']):
            X_engineered['solids_conductivity_ratio'] = X['solids'] / (X['conductivity'] + 1e-8)
        
        # Turbidity interactions
        if all(col in X.columns for col in ['turbidity', 'solids']):
            X_engineered['turbidity_solids_product'] = X['turbidity'] * X['solids']
        
        # Organic carbon interactions
        if all(col in X.columns for col in ['organic_carbon', 'trihalomethanes']):
            X_engineered['organic_trihalomethanes_ratio'] = X['organic_carbon'] / (X['trihalomethanes'] + 1e-8)
        
        # Polynomial features for key parameters
        if 'ph' in X.columns:
            X_engineered['ph_squared'] = X['ph'] ** 2
            X_engineered['ph_log'] = np.log(np.abs(X['ph']) + 1e-8)
        
        if 'hardness' in X.columns:
            X_engineered['hardness_squared'] = X['hardness'] ** 2
            X_engineered['hardness_sqrt'] = np.sqrt(np.abs(X['hardness']))
        
        # Aggregated features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_engineered['mean_chemical'] = X[numeric_cols].mean(axis=1)
        X_engineered['std_chemical'] = X[numeric_cols].std(axis=1)
        
        # Store engineered feature names
        self.engineered_features = list(set(X_engineered.columns) - set(X.columns))
        
        return X_engineered
    
    def create_physically_constrained_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create features based on physical constraints"""
        X_constrained = X.copy()
        
        # pH deviation from optimal range (6.5-8.5)
        ph_optimal_min, ph_optimal_max = 6.5, 8.5
        if 'ph' in X.columns:
            ph_values = X['ph'].astype(float)
        elif 'pH' in X.columns:
            ph_values = X['pH'].astype(float)
        else:
            ph_values = pd.Series(np.zeros(len(X)), index=X.index)
        
        # Convert to numpy array explicitly
        ph_values = np.array(ph_values, dtype=float)
        
        ph_deviation = np.maximum(
            ph_optimal_min - ph_values, 
            ph_values - ph_optimal_max, 
            0
        )
        X_constrained['ph_deviation'] = ph_deviation
            
        # pH category
        if 'ph' in X.columns:
            X_constrained['ph_category'] = pd.cut(
                X['ph'], 
                bins=[0, 6.5, 8.5, 14], 
                labels=['acidic', 'optimal', 'alkaline']
            ).astype('category')
        
        # Turbidity-based features
        if 'turbidity' in X.columns:
            # Turbidity above WHO limit (5 NTU)
            turbidity_limit = 5.0
            turbidity_values = X['turbidity'].astype(float)
            # Convert to numpy array explicitly
            turbidity_values = np.array(turbidity_values, dtype=float)
            turbidity_excess = np.maximum(turbidity_values - turbidity_limit, 0)
            X_constrained['turbidity_excess'] = turbidity_excess
            
            # Turbidity category
            X_constrained['turbidity_category'] = pd.cut(
                X['turbidity'],
                bins=[0, 1, 5, float('inf')],
                labels=['excellent', 'acceptable', 'poor']
            ).astype('category')
        
        # Sulfate-based features
        if 'sulfate' in X.columns:
            # Sulfate deviation from WHO limit (250 mg/L)
            sulfate_limit = 250.0
            sulfate_values = X['sulfate'].astype(float)
            # Convert to numpy array explicitly
            sulfate_values = np.array(sulfate_values, dtype=float)
            sulfate_excess = np.maximum(sulfate_values - sulfate_limit, 0)
            X_constrained['sulfate_excess'] = sulfate_excess
        
        # Chloramines-based features
        if 'chloramines' in X.columns:
            # Chloramines deviation from WHO limit (4 mg/L)
            chloramines_limit = 4.0
            chloramines_values = X['chloramines'].astype(float)
            # Convert to numpy array explicitly
            chloramines_values = np.array(chloramines_values, dtype=float)
            chloramines_excess = np.maximum(chloramines_values - chloramines_limit, 0)
            X_constrained['chloramines_excess'] = chloramines_excess
        
        return X_constrained
    
    def create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        X_stats = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics (if data has temporal component)
        if len(X) > 10:
            window_size = min(10, len(X) // 4)
            for col in numeric_cols:
                X_stats[f'{col}_rolling_mean'] = X[col].rolling(
                    window=window_size, min_periods=1
                ).mean()
                X_stats[f'{col}_rolling_std'] = X[col].rolling(
                    window=window_size, min_periods=1
                ).std()
        
        # Z-scores
        for col in numeric_cols:
            X_stats[f'{col}_zscore'] = stats.zscore(X[col])
        
        # Percentile ranks
        for col in numeric_cols:
            X_stats[f'{col}_percentile'] = X[col].rank(pct=True)
        
        return X_stats
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Step 1: Create interaction features
        X_engineered = self.create_interaction_features(X)
        print(f"Created {len(self.engineered_features)} interaction features")
        
        # Step 2: Create physically constrained features
        X_constrained = self.create_physically_constrained_features(X_engineered)
        
        # Step 3: Create statistical features
        X_final = self.create_statistical_features(X_constrained)
        
        print(f"Final feature count: {X_final.shape[1]} (original: {X.shape[1]})")
        
        self.feature_names = X_final.columns.tolist()
        return X_final


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline combining Robust PCA and Feature Engineering
    """
    
    def __init__(self, apply_pca: bool = True, pca_components: float = 0.95):
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.robust_pca = RobustPCATransformer(n_components=pca_components) if apply_pca else None
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'PreprocessingPipeline':
        """Fit preprocessing pipeline"""
        print("Fitting preprocessing pipeline...")
        
        # Feature engineering
        X_engineered = self.feature_engineer.fit_transform(X)
        
        # Robust PCA (if enabled)
        if self.apply_pca:
            self.robust_pca.fit(X_engineered)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("PreprocessingPipeline not fitted yet")
        
        # Feature engineering
        X_engineered = self.feature_engineer.create_interaction_features(X)
        X_constrained = self.feature_engineer.create_physically_constrained_features(X_engineered)
        X_final = self.feature_engineer.create_statistical_features(X_constrained)
        
        # Robust PCA (if enabled)
        if self.apply_pca:
            X_final = self.robust_pca.transform(X_final)
        
        return X_final
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def get_feature_importance_report(self) -> dict:
        """Get feature importance and engineering report"""
        if not self.is_fitted:
            raise ValueError("PreprocessingPipeline not fitted yet")
        
        report = {
            'original_features': len(self.feature_engineer.feature_names) - len(self.feature_engineer.engineered_features),
            'engineered_features': len(self.feature_engineer.engineered_features),
            'total_features': len(self.feature_engineer.feature_names),
            'engineered_feature_list': self.feature_engineer.engineered_features,
            'pca_applied': self.apply_pca,
            'pca_components': self.robust_pca.pca.n_components_ if self.apply_pca else None
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, _ = make_classification(n_samples=1000, n_features=9, random_state=42)
    
    # Create realistic water quality feature names
    feature_names = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
                    'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
    X = pd.DataFrame(X, columns=feature_names)
    
    print("Original data shape:", X.shape)
    print("Original features:", X.columns.tolist())
    
    # Apply preprocessing
    preprocessor = PreprocessingPipeline(apply_pca=True, pca_components=0.95)
    X_processed = preprocessor.fit_transform(X)
    
    print("\nProcessed data shape:", X_processed.shape)
    
    # Feature importance report
    report = preprocessor.get_feature_importance_report()
    print("\nFeature Engineering Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
