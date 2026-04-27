# Technical Documentation - System Architecture

## 🎯 SYSTEM OVERVIEW

**Explainable Federated Physics-Aware Deep Learning Framework for Scalable Water Potability Prediction**

### **ARCHITECTURE PHILOSOPHY**
- **Modular Design**: Clean separation of concerns
- **Domain Integration**: WHO water quality standards embedded
- **Advanced Preprocessing**: State-of-the-art data handling
- **Comprehensive Evaluation**: Multi-metric assessment
- **Production Ready**: Robust error handling and logging

---

## 🏗️ SYSTEM ARCHITECTURE

### **HIGH-LEVEL ARCHITECTURE**
```
┌─────────────────────────────────────────────────────────────┐
│                    WATER POTABILITY PREDICTION               │
│                      FRAMEWORK ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA INGESTION LAYER                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Raw Data    │  │ Validation  │  │ Quality Check        │ │
│  │ (CSV/JSON)  │  │ (Schema)    │  │ (Missing/Outliers)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ GAIN        │  │ Feature     │  │ Ensemble             │ │
│  │ Imputation  │  │ Engineering │  │ Normalization        │ │
│  │ (Neural)    │  │ (WHO)       │  │ (3 Methods)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     MODELING LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Classical   │  │ Deep        │  │ Ensemble             │ │
│  │ Models      │  │ Learning    │  │ Methods              │ │
│  │ (6 Types)   │  │ (TabTrans)  │  │ (Voting/Stacking)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY LAYER                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ SHAP        │  │ Counter-    │  │ Physics-Aware        │ │
│  │ Values      │  │ factuals    │  │ Explanations         │ │
│  │ (Global)    │  │ (What-if)   │  │ (WHO Constraints)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Multi-      │  │ Cross       │  │ Statistical          │ │
│  │ Metric      │  │ Validation  │  │ Analysis             │ │
│  │ (5 Types)   │  │ (5-Fold)    │  │ (Significance)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Federated   │  │ API         │  │ Monitoring           │ │
│  │ Learning    │  │ Endpoints   │  │ (Performance)        │ │
│  │ (Multi-     │  │ (REST)      │  │ (Drift Detection)    │ │
│  │ client)     │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 COMPONENT ARCHITECTURE

### **1. DATA CLEANING MODULE**

#### **GAIN IMPUTATION SYSTEM**
```python
class GAINImputer:
    """
    Generative Adversarial Imputation Networks
    Architecture: Generator + Discriminator
    """
    
    def __init__(self, hidden_dim=64, epochs=500):
        self.generator = Generator(hidden_dim)
        self.discriminator = Discriminator(hidden_dim)
        self.epochs = epochs
    
    def impute(self, data_with_missing):
        # Training loop
        for epoch in range(self.epochs):
            # Generator forward pass
            generated_data = self.generator(missing_mask)
            
            # Discriminator evaluation
            real_loss = self.discriminator(real_data)
            fake_loss = self.discriminator(generated_data)
            
            # Adversarial training
            self.train_generator(generated_data, real_loss)
            self.train_discriminator(real_loss, fake_loss)
        
        return self.generator.predict(data_with_missing)
```

#### **OUTLIER DETECTION SYSTEM**
```python
class OutlierDetector:
    """
    Isolation Forest for outlier detection
    Contamination: 10% (configurable)
    """
    
    def __init__(self, contamination=0.1):
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
    
    def detect_outliers(self, data):
        outlier_labels = self.iso_forest.fit_predict(data)
        outlier_mask = outlier_labels == -1
        return outlier_mask
```

### **2. FEATURE ENGINEERING MODULE**

#### **PHYSICS-AWARE FEATURE ENGINEERING**
```python
class PhysicsAwareFeatureEngineer:
    """
    WHO Water Quality Standards Integration
    """
    
    WHO_STANDARDS = {
        'ph': {'min': 6.5, 'max': 8.5},
        'turbidity': {'max': 5.0},  # NTU
        'sulfate': {'max': 250.0},  # mg/L
        'chloramines': {'max': 4.0}  # mg/L
    }
    
    def create_physics_features(self, df):
        # pH-based features
        df['ph_deviation'] = self._calculate_ph_deviation(df['ph'])
        df['ph_squared'] = df['ph'] ** 2
        df['ph_log'] = np.log(np.abs(df['ph']) + 1e-8)
        
        # WHO constraint violations
        df['turbidity_excess'] = np.maximum(df['turbidity'] - 5.0, 0)
        df['sulfate_excess'] = np.maximum(df['sulfate'] - 250.0, 0)
        df['chloramines_excess'] = np.maximum(df['chloramines'] - 4.0, 0)
        
        # Chemical interaction features
        df['hardness_ph_ratio'] = df['hardness'] / (df['ph'] + 1e-8)
        df['solids_conductivity_ratio'] = df['solids'] / (df['conductivity'] + 1e-8)
        
        # Aggregate features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['mean_chemical'] = df[numeric_cols].mean(axis=1)
        df['std_chemical'] = df[numeric_cols].std(axis=1)
        
        return df
```

### **3. NORMALIZATION MODULE**

#### **ENSEMBLE NORMALIZATION SYSTEM**
```python
class EnsembleNormalizer:
    """
    Ensemble Normalization: Quantile + Copula + Robust
    """
    
    def __init__(self, weights=[0.4, 0.3, 0.3]):
        self.weights = weights
        self.quantile_transformer = QuantileTransformer()
        self.copula_transformer = CopulaTransformer()
        self.robust_scaler = RobustScaler()
    
    def fit_transform(self, data):
        # Apply individual transformations
        quantile_data = self.quantile_transformer.fit_transform(data)
        copula_data = self.copula_transformer.fit_transform(data)
        robust_data = self.robust_scaler.fit_transform(data)
        
        # Weighted ensemble
        ensemble_data = (
            self.weights[0] * quantile_data +
            self.weights[1] * copula_data +
            self.weights[2] * robust_data
        )
        
        return ensemble_data
```

### **4. MODEL ARCHITECTURE**

#### **CLASSICAL MODELS SUITE**
```python
class ModelSuite:
    """
    Suite of 5 models (4 classical + 1 deep learning)
    """
    
    def __init__(self):
        self.models = {
            'TabTransformer': RealWaterPotabilityPredictor(
                input_dim=13, d_model=64, n_heads=8, n_layers=4
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            )
        }
    
    def train_all(self, X_train, y_train):
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            results[name] = model
        return results
```

#### **TABTRANSFORMER ARCHITECTURE**
```python
class TabTransformer(nn.Module):
    """
    Transformer Architecture for Tabular Data
    """
    
    def __init__(self, input_dim, embed_dim=64, num_heads=8, num_layers=4):
        super().__init__()
        
        # Feature embedding layer
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, 1)
        
        # Physics-aware loss
        self.physics_loss = PhysicsAwareLoss()
    
    def forward(self, x):
        # Feature embedding
        x = self.feature_embedding(x)
        x = self.pos_encoding(x)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Classification
        logits = self.classifier(x)
        return logits
```

### **5. EXPLAINABLE AI MODULE**

#### **SHAP-BASED EXPLANATIONS**
```python
class SHAPExplainer:
    """
    SHAP-based model explanations
    """
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def fit_explainer(self, X_background):
        # Model-specific explainer selection
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, 'coef_'):
            # Linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            # Kernel explainer for any model
            self.explainer = shap.KernelExplainer(self.model, X_background)
    
    def explain_instance(self, instance):
        shap_values = self.explainer.shap_values(instance)
        return {
            'feature_importance': list(zip(self.feature_names, shap_values)),
            'prediction': self.model.predict(instance),
            'probability': self.model.predict_proba(instance)
        }
```

#### **PHYSICS-AWARE EXPLANATIONS**
```python
class PhysicsAwareExplainer:
    """
    WHO water quality constraint explanations
    """
    
    def __init__(self):
        self.who_standards = WHO_STANDARDS
    
    def explain_physics_constraints(self, sample):
        explanations = {}
        
        for param, standards in self.who_standards.items():
            value = sample[param]
            
            if 'min' in standards and 'max' in standards:
                status = self._evaluate_range(value, standards)
            elif 'max' in standards:
                status = self._evaluate_maximum(value, standards['max'])
            
            explanations[param] = {
                'value': value,
                'standard': standards,
                'status': status,
                'violation': value > standards.get('max', float('inf'))
            }
        
        return explanations
```

---

## 🗄️ DATA FLOW ARCHITECTURE

### **PIPELINE DATA FLOW**
```
RAW DATA INPUT
    │
    ▼
┌─────────────────────┐
│   DATA VALIDATION   │ ← Schema validation, type checking
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   QUALITY CHECK     │ ← Missing value analysis, outlier detection
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   GAIN IMPUTATION   │ ← Neural network missing data handling
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ FEATURE ENGINEERING │ ← WHO standards, interactions, aggregates
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ ENSEMBLE NORMALIZE  │ ← Quantile + Copula + Robust scaling
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   MODEL TRAINING    │ ← 6 classical models + TabTransformer
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   MODEL EVALUATION  │ ← Multi-metric assessment, validation
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   EXPLAINABILITY    │ ← SHAP values, physics-aware explanations
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   REPORTING         │ ← Visualizations, HTML, PDF, text reports
└─────────────────────┘
```

### **DATA SCHEMA**
```python
DATA_SCHEMA = {
    'input_features': {
        'ph': 'float64',           # pH level (0-14)
        'hardness': 'float64',     # Water hardness (mg/L)
        'solids': 'float64',       # Total dissolved solids (ppm)
        'chloramines': 'float64',  # Chloramines (ppm)
        'sulfate': 'float64',      # Sulfates (mg/L)
        'conductivity': 'float64', # Electrical conductivity (μS/cm)
        'organic_carbon': 'float64',# Organic carbon (ppm)
        'trihalomethanes': 'float64', # Trihalomethanes (μg/L)
        'turbidity': 'float64'     # Turbidity (NTU)
    },
    'target': {
        'potability': 'int64'      # 0=non-potable, 1=potable
    },
    'engineered_features': {
        'ph_deviation': 'float64',
        'ph_squared': 'float64',
        'ph_log': 'float64',
        'hardness_ph_ratio': 'float64',
        'solids_conductivity_ratio': 'float64',
        'turbidity_excess': 'float64',
        'sulfate_excess': 'float64',
        'chloramines_excess': 'float64',
        'mean_chemical': 'float64',
        'std_chemical': 'float64'
    }
}
```

---

## 🔧 CONFIGURATION ARCHITECTURE

### **SYSTEM CONFIGURATION**
```python
SYSTEM_CONFIG = {
    'data_cleaning': {
        'gain_params': {
            'hidden_dim': 64,
            'epochs': 500,
            'batch_size': 128
        },
        'outlier_params': {
            'contamination': 0.1,
            'random_state': 42
        }
    },
    'feature_engineering': {
        'who_standards': True,
        'interactions': True,
        'non_linear': True,
        'aggregates': True
    },
    'normalization': {
        'method': 'ensemble',
        'methods': ['quantile', 'copula', 'robust'],
        'weights': [0.4, 0.3, 0.3]
    },
    'models': {
        'deep_learning': {
            'tabtransformer': {
                'input_dim': 13,
                'd_model': 64,
                'n_heads': 8,
                'n_layers': 4,
                'dropout': 0.1
            }
        },
        'classical': {
            'naive_bayes': {},
            'random_forest': {'n_estimators': 100},
            'decision_tree': {'max_depth': None},
            'knn': {'n_neighbors': 5}
        }
    },
    'xai': {
        'method': 'attention',
        'physics_aware': True,
        'counterfactuals': True
    },
    'federated': {
        'num_clients': 4,
        'aggregation': 'fedavg',
        'rounds': 10,
        'client_fraction': 0.8
    }
}
```

---

## 🚀 DEPLOYMENT ARCHITECTURE

### **PRODUCTION DEPLOYMENT**
```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION ENVIRONMENT                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API GATEWAY LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Request     │  │ Rate        │  │ Authentication       │ │
│  │ Validation  │  │ Limiting    │  │ & Authorization      │ │
│  │ (Schema)    │  │ (1000/min)  │  │ (JWT/OAuth)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREDICTION SERVICE                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Model       │  │ Batch       │  │ Real-time            │ │
│  │ Loading     │  │ Processing  │  │ Inference            │ │
│  │ (Memory)    │  │ (Queue)     │  │ (GPU/CPU)            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  MONITORING LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Performance │  │ Model       │  │ Data                 │ │
│  │ Metrics     │  │ Drift       │  │ Quality              │ │
│  │ (Latency)   │  │ Detection   │  │ (Missing/Outliers)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### **FEDERATED LEARNING ARCHITECTURE**
```
┌─────────────────────────────────────────────────────────────┐
│                  FEDERATED SERVER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Model       │  │ Aggregation │  │ Client               │ │
│  │ Storage     │  │ Service     │  │ Management           │ │
│  │ (Global)    │  │ (FedAvg)    │  │ (Registration)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLIENTS (Water Utilities)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Client 1    │  │ Client 2    │  │ Client N             │ │
│  │ (Urban)     │  │ (Rural)     │  │ (Industrial)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 PERFORMANCE ARCHITECTURE

### **SCALABILITY DESIGN**
```python
class ScalableArchitecture:
    """
    Scalability considerations for water potability prediction
    """
    
    PERFORMANCE_TARGETS = {
        'prediction_latency': '<100ms',
        'batch_processing': '>1000 samples/sec',
        'memory_usage': '<4GB RAM',
        'model_loading': '<5 seconds',
        'api_response': '<200ms'
    }
    
    SCALABILITY_FEATURES = {
        'horizontal_scaling': 'Multiple API instances',
        'model_caching': 'Pre-loaded models in memory',
        'batch_processing': 'Queue-based bulk predictions',
        'load_balancing': 'Round-robin distribution',
        'caching_layer': 'Redis for frequent predictions'
    }
```

### **MONITORING ARCHITECTURE**
```python
class MonitoringSystem:
    """
    Comprehensive monitoring for production deployment
    """
    
    METRICS = {
        'model_performance': ['accuracy', 'precision', 'recall', 'f1'],
        'system_performance': ['cpu_usage', 'memory_usage', 'disk_io'],
        'api_metrics': ['request_count', 'response_time', 'error_rate'],
        'data_quality': ['missing_values', 'outliers', 'drift_detection']
    }
    
    ALERTS = {
        'performance_degradation': 'accuracy_drop > 5%',
        'system_overload': 'cpu_usage > 80%',
        'data_drift': 'feature_distribution_change > 10%',
        'api_errors': 'error_rate > 5%'
    }
```

---

## 🔧 IMPLEMENTATION DETAILS

### **CODE ORGANIZATION**
```
water_potability_prediction/
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py          # GAIN imputation, outlier detection
│   │   ├── preprocessing.py          # Feature engineering
│   │   └── normalization.py          # Ensemble normalization
│   ├── models/
│   │   ├── classical_models.py       # 6 classical ML models
│   │   ├── tabtransformer.py         # Deep learning architecture
│   │   └── ensemble_methods.py       # Voting/stacking ensembles
│   ├── xai/
│   │   ├── explainable_ai.py         # SHAP explanations
│   │   ├── physics_explainer.py      # WHO constraint explanations
│   │   └── counterfactuals.py        # What-if scenarios
│   ├── evaluation/
│   │   ├── evaluation.py             # Multi-metric evaluation
│   │   ├── cross_validation.py       # 5-fold CV framework
│   │   └── statistical_tests.py     # Significance testing
│   ├── federated/
│   │   ├── federated_learning.py     # FedAvg implementation
│   │   ├── client_simulation.py      # Multi-utility simulation
│   │   └── aggregation_methods.py    # FedAvg, FedProx
│   └── deployment/
│       ├── api_server.py             # REST API endpoints
│       ├── batch_processor.py        # Bulk prediction service
│       └── monitoring.py             # Performance monitoring
├── docs/
│   ├── comprehensive_summary.md      # System overview
│   ├── model_comparison_analysis.md  # Detailed model analysis
│   ├── technical_architecture.md     # This document
│   └── deployment_guide.md          # Production deployment
├── results/
│   ├── summary.txt                   # Current results
│   ├── model_comparison_report.txt   # Model analysis
│   └── visualizations/               # Charts and graphs
├── tests/
│   ├── unit_tests/                   # Component testing
│   ├── integration_tests/            # Pipeline testing
│   └── performance_tests/           # Load testing
├── main.py                          # Main pipeline execution
├── requirements.txt                  # Dependencies
└── README.md                        # Quick start guide
```

---

## 🎯 CONCLUSION

### **ARCHITECTURE STRENGTHS**
✅ **Modular Design**: Clean separation of concerns  
✅ **Domain Integration**: WHO water quality standards embedded  
✅ **Advanced Preprocessing**: State-of-the-art data handling  
✅ **Comprehensive Evaluation**: Multi-metric assessment  
✅ **Explainability**: SHAP + physics-aware explanations  
✅ **Scalability**: Federated learning ready  
✅ **Production Ready**: Monitoring, API, deployment support  

### **TECHNICAL INNOVATIONS**
🔬 **GAIN Imputation**: Neural network missing data handling  
🔬 **Ensemble Normalization**: Quantile + Copula + Robust  
🔬 **Physics-Aware Features**: WHO standards integration  
🔬 **TabTransformer**: Deep learning for tabular water data  
🔬 **Federated Architecture**: Multi-utility deployment  

### **SYSTEM MATURITY**
🎯 **Development**: Complete implementation  
🎯 **Testing**: Comprehensive evaluation completed  
🎯 **Documentation**: Full technical documentation  
🎯 **Deployment**: Production architecture ready  
🎯 **Maintenance**: Monitoring and update framework  

---

*Architecture Version: 1.0*  
*Last Updated: Current pipeline execution*  
*System Status: Production Ready*  
*Documentation: Complete and Comprehensive*
