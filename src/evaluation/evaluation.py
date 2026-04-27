"""
Evaluation Module for Water Potability Prediction
Implements comprehensive model evaluation and comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, model_name: str = "TabTransformer"):
        self.model_name = model_name
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      X_train: Optional[pd.DataFrame] = None,
                      y_train: Optional[pd.Series] = None) -> Dict:
        """
        Comprehensive model evaluation
        """
        print(f"Evaluating {self.model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            try:
                y_pred_proba = model.predict(X_test)
            except:
                print("Could not get prediction probabilities")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # ROC AUC (if probabilities available)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred)
        
        # Store results
        self.evaluation_results = metrics
        
        return metrics
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5) -> Dict:
        """
        Stratified K-fold cross-validation
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        # Create stratified K-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Manual cross-validation to get detailed metrics
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model on fold
            model_copy = self._copy_model(model)
            model_copy.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            y_pred_fold = model_copy.predict(X_val_fold)
            
            # Calculate metrics
            cv_metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
            cv_metrics['precision'].append(precision_score(y_val_fold, y_pred_fold, average='weighted'))
            cv_metrics['recall'].append(recall_score(y_val_fold, y_pred_fold, average='weighted'))
            cv_metrics['f1'].append(f1_score(y_val_fold, y_pred_fold, average='weighted'))
        
        # Calculate mean and std
        cv_results = {}
        for metric in cv_metrics:
            cv_results[f'{metric}_mean'] = np.mean(cv_metrics[metric])
            cv_results[f'{metric}_std'] = np.std(cv_metrics[metric])
        
        return cv_results
    
    def _copy_model(self, model):
        """Create a copy of the model for cross-validation"""
        if hasattr(model, 'get_params'):
            return type(model)(**model.get_params())
        else:
            # For custom models, return the same instance
            # Note: This is a simplified approach
            return model
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['Not Potable', 'Potable']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        """Plot calibration curve"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{self.model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve - {self.model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class ModelComparator:
    """
    Model comparison framework
    """
    
    def __init__(self):
        self.models = {}
        self.evaluations = {}
        self.comparison_results = {}
        
    def add_model(self, name: str, model, X_train: pd.DataFrame, y_train: pd.Series):
        """Add a model to comparison"""
        print(f"Adding {name} to comparison...")
        
        # Train model
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
        
        self.models[name] = model
        
        # Evaluate model
        evaluator = ModelEvaluator(name)
        metrics = evaluator.evaluate_model(model, X_train, y_train)
        self.evaluations[name] = metrics
        
        print(f"{name} added successfully!")
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Compare all models"""
        print("Comparing models...")
        
        comparison_data = []
        
        for name, model in self.models.items():
            # Get predictions
            y_pred = model.predict(X_test)
            
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
            
            comparison_data.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score
        comparison_df = comparison_df.sort_values('F1', ascending=False)
        
        self.comparison_results = comparison_df
        
        return comparison_df
    
    def create_baseline_models(self) -> Dict:
        """Create baseline models for comparison"""
        baseline_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        return baseline_models
    
    def plot_model_comparison(self, metrics: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1']):
        """Plot model comparison"""
        if not self.comparison_results or len(self.comparison_results) == 0:
            print("No comparison results available. Run compare_models() first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if metric in self.comparison_results.columns:
                ax = axes[i]
                
                # Sort by current metric
                sorted_df = self.comparison_results.sort_values(metric, ascending=True)
                
                # Create horizontal bar plot
                bars = ax.barh(sorted_df['Model'], sorted_df[metric], 
                              color=plt.cm.viridis(np.linspace(0, 1, len(sorted_df))))
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center')
                
                ax.set_title(f'{metric} Comparison')
                ax.set_xlabel(metric)
                ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_comparison(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
            except:
                print(f"Could not plot ROC curve for {name}")
        
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def generate_comparison_report(self, output_file: str = "model_comparison_report.html") -> str:
        """Generate comprehensive comparison report"""
        if not self.comparison_results or len(self.comparison_results) == 0:
            print("No comparison results available. Run compare_models() first.")
            return ""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Water Potability Prediction - Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>Water Potability Prediction - Model Comparison Report</h1>
            
            <div class="section">
                <h2>Model Performance Comparison</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
            """
        
        # Add ROC-AUC if available
        if 'ROC-AUC' in self.comparison_results.columns:
            html_content += "<th>ROC-AUC</th>"
        
        html_content += "</tr>"
        
        # Find best model for each metric
        best_metrics = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
            if metric in self.comparison_results.columns:
                best_metrics[metric] = self.comparison_results[metric].max()
        
        if 'ROC-AUC' in self.comparison_results.columns:
            best_metrics['ROC-AUC'] = self.comparison_results['ROC-AUC'].max()
        
        # Add table rows
        for _, row in self.comparison_results.iterrows():
            model_name = row['Model']
            
            # Check if this is the best model for any metric
            is_best = False
            for metric, best_value in best_metrics.items():
                if metric in row and abs(row[metric] - best_value) < 1e-6:
                    is_best = True
                    break
            
            row_class = 'best' if is_best else ''
            
            html_content += f'<tr class="{row_class}">'
            html_content += f'<td><strong>{model_name}</strong></td>'
            
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                if metric in row:
                    html_content += f'<td>{row[metric]:.4f}</td>'
            
            if 'ROC-AUC' in row:
                html_content += f'<td>{row["ROC-AUC"]:.4f}</td>'
            
            html_content += '</tr>'
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>This report compares the performance of different machine learning models 
                for water potability prediction. The best performing model is highlighted in green.</p>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Model comparison report saved to {output_file}")
        return output_file


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=9, n_informative=7, 
                              n_redundant=2, random_state=42)
    
    feature_names = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
                    'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create model comparator
    comparator = ModelComparator()
    
    # Create baseline models
    baseline_models = comparator.create_baseline_models()
    
    # Add models to comparison
    for name, model in baseline_models.items():
        try:
            comparator.add_model(name, model, X_train, y_train)
        except Exception as e:
            print(f"Error adding {name}: {e}")
    
    # Compare models
    comparison_results = comparator.compare_models(X_test, y_test)
    
    print("\nModel Comparison Results:")
    print(comparison_results)
    
    # Plot comparisons
    comparator.plot_model_comparison()
    comparator.plot_roc_comparison(X_test, y_test)
    
    # Generate report
    comparator.generate_comparison_report()
