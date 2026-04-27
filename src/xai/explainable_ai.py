"""
REAL Explainable AI Implementation with SHAP and Counterfactuals
Advanced XAI for water potability prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch

class ExplainableAI:
    """Real XAI system with SHAP, counterfactuals, and attention visualization"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit_explainer(self, X_train: pd.DataFrame):
        """Fit SHAP explainer on training data"""
        print("🧠 Fitting SHAP explainer...")
        
        if hasattr(self.model, 'predict_proba'):
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ Using TreeExplainer for tree-based model")
        else:
            # Linear models
            self.explainer = shap.LinearExplainer(self.model, X_train)
            print("✅ Using LinearExplainer for linear model")
            
        print("✅ SHAP explainer fitted successfully")
    
    def explain_prediction(self, instance: pd.Series, background_data: pd.DataFrame) -> Dict:
        """Explain a single prediction with comprehensive XAI"""
        print("🔍 Generating comprehensive explanation...")
        
        # SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Feature importance ranking
        feature_importance = dict(zip(self.feature_names, shap_values))
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Physics-aware analysis
        physics_analysis = self._analyze_physics_constraints(instance)
        
        # Counterfactual explanations
        counterfactuals = self._generate_counterfactuals(instance, background_data)
        
        # Water chemistry insights
        chemistry_insights = self._analyze_water_chemistry(instance)
        
        explanation = {
            'prediction': self.model.predict([instance])[0],
            'probability': self.model.predict_proba([instance])[0, 1],
            'shap_values': shap_values,
            'feature_importance': sorted_features,
            'physics_analysis': physics_analysis,
            'counterfactuals': counterfactuals,
            'chemistry_insights': chemistry_insights,
            'recommendations': self._generate_recommendations(instance, physics_analysis)
        }
        
        return explanation
    
    def _analyze_physics_constraints(self, instance: pd.Series) -> Dict:
        """Analyze WHO water quality constraints"""
        print("🔬 Analyzing physics constraints...")
        
        constraints = {}
        
        # pH analysis
        ph = instance.get('ph', 7.0)
        if 6.5 <= ph <= 8.5:
            constraints['ph'] = {'status': 'COMPLIANT', 'range': '6.5-8.5', 'value': ph}
        else:
            constraints['ph'] = {'status': 'VIOLATION', 'range': '6.5-8.5', 'value': ph}
        
        # Turbidity analysis
        turbidity = instance.get('Turbidity', 5.0)
        if turbidity < 5.0:
            constraints['turbidity'] = {'status': 'COMPLIANT', 'limit': '<5 NTU', 'value': turbidity}
        else:
            constraints['turbidity'] = {'status': 'VIOLATION', 'limit': '<5 NTU', 'value': turbidity}
        
        # Sulfate analysis
        sulfate = instance.get('Sulfate', 200.0)
        if sulfate < 250.0:
            constraints['sulfate'] = {'status': 'COMPLIANT', 'limit': '<250 mg/L', 'value': sulfate}
        else:
            constraints['sulfate'] = {'status': 'VIOLATION', 'limit': '<250 mg/L', 'value': sulfate}
        
        # Chloramines analysis
        chloramines = instance.get('Chloramines', 4.0)
        if chloramines < 4.0:
            constraints['chloramines'] = {'status': 'COMPLIANT', 'limit': '<4 mg/L', 'value': chloramines}
        else:
            constraints['chloramines'] = {'status': 'VIOLATION', 'limit': '<4 mg/L', 'value': chloramines}
        
        # Overall compliance score
        violations = sum(1 for c in constraints.values() if c['status'] == 'VIOLATION')
        compliance_score = (len(constraints) - violations) / len(constraints)
        
        constraints['overall_compliance'] = {
            'score': compliance_score,
            'violations': violations,
            'status': 'EXCELLENT' if violations == 0 else 'GOOD' if violations == 1 else 'POOR'
        }
        
        return constraints
    
    def _generate_counterfactuals(self, instance: pd.Series, background_data: pd.DataFrame, n_counterfactuals: int = 3) -> List[Dict]:
        """Generate counterfactual explanations"""
        print("⚡ Generating counterfactual explanations...")
        
        counterfactuals = []
        current_pred = self.model.predict([instance])[0]
        
        # Find similar instances from background
        similar_instances = background_data.copy()
        for col in instance.index:
            similar_instances = similar_instances[
                (similar_instances[col] >= instance[col] * 0.8) & 
                (similar_instances[col] <= instance[col] * 1.2)
            ]
        
        if len(similar_instances) == 0:
            similar_instances = background_data.sample(min(n_counterfactuals, len(background_data)))
        
        for _, similar in similar_instances.head(n_counterfactuals).iterrows():
            # Create counterfactual by modifying key features
            counterfactual = similar.copy()
            
            # Modify pH to be optimal
            if 'ph' in counterfactual:
                counterfactual['ph'] = 7.0  # Optimal pH
            
            # Modify turbidity to be compliant
            if 'Turbidity' in counterfactual:
                counterfactual['Turbidity'] = min(counterfactual['Turbidity'], 4.5)
            
            # Modify sulfate to be safe
            if 'Sulfate' in counterfactual:
                counterfactual['Sulfate'] = min(counterfactual['Sulfate'], 200.0)
            
            # Check if this changes prediction
            new_pred = self.model.predict([counterfactual])[0]
            
            counterfactuals.append({
                'original': instance.to_dict(),
                'counterfactual': counterfactual.to_dict(),
                'original_prediction': current_pred,
                'counterfactual_prediction': new_pred,
                'prediction_changed': new_pred != current_pred,
                'changes_made': self._get_changes(instance, counterfactual)
            })
        
        return counterfactuals
    
    def _analyze_water_chemistry(self, instance: pd.Series) -> Dict:
        """Analyze water chemistry relationships"""
        print("🧪 Analyzing water chemistry...")
        
        chemistry = {}
        
        # pH and hardness relationship
        ph = instance.get('ph', 7.0)
        hardness = instance.get('Hardness', 200.0)
        
        if ph < 6.5 and hardness > 300:
            chemistry['ph_hardness'] = {
                'relationship': 'Acidic water tends to have higher hardness',
                'status': 'EXPECTED PATTERN'
            }
        elif ph > 8.5 and hardness < 100:
            chemistry['ph_hardness'] = {
                'relationship': 'Alkaline water tends to have lower hardness',
                'status': 'EXPECTED PATTERN'
            }
        else:
            chemistry['ph_hardness'] = {
                'relationship': 'Normal pH-hardness relationship',
                'status': 'WITHIN EXPECTED RANGE'
            }
        
        # Solids and conductivity relationship
        solids = instance.get('Solids', 300.0)
        conductivity = instance.get('Conductivity', 15.0)
        
        if solids > 500 and conductivity > 20:
            chemistry['solids_conductivity'] = {
                'relationship': 'High dissolved solids increase conductivity',
                'status': 'EXPECTED PATTERN'
            }
        else:
            chemistry['solids_conductivity'] = {
                'relationship': 'Normal solids-conductivity relationship',
                'status': 'WITHIN EXPECTED RANGE'
            }
        
        # Chloramines and organic carbon relationship
        chloramines = instance.get('Chloramines', 3.5)
        organic_carbon = instance.get('Organic_carbon', 80.0)
        
        if chloramines > 3.0 and organic_carbon > 100:
            chemistry['chloramines_organic'] = {
                'relationship': 'High chloramines often accompany high organic carbon',
                'status': 'EXPECTED PATTERN'
            }
        else:
            chemistry['chloramines_organic'] = {
                'relationship': 'Normal chloramines-organic carbon relationship',
                'status': 'WITHIN EXPECTED RANGE'
            }
        
        return chemistry
    
    def _generate_recommendations(self, instance: pd.Series, physics_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # pH recommendations
        if 'ph' in physics_analysis:
            ph_status = physics_analysis['ph']['status']
            if ph_status == 'VIOLATION':
                ph_value = physics_analysis['ph']['value']
                if ph_value < 6.5:
                    recommendations.append(f"🔧 Increase pH from {ph_value:.2f} to 6.5-8.5 range using lime treatment")
                else:
                    recommendations.append(f"🔧 Decrease pH from {ph_value:.2f} to 6.5-8.5 range using acid treatment")
        
        # Turbidity recommendations
        if 'turbidity' in physics_analysis:
            turbidity_status = physics_analysis['turbidity']['status']
            if turbidity_status == 'VIOLATION':
                turbidity_value = physics_analysis['turbidity']['value']
                recommendations.append(f"🔧 Reduce turbidity from {turbidity_value:.2f} NTU to <5 NTU using filtration")
        
        # Sulfate recommendations
        if 'sulfate' in physics_analysis:
            sulfate_status = physics_analysis['sulfate']['status']
            if sulfate_status == 'VIOLATION':
                sulfate_value = physics_analysis['sulfate']['value']
                recommendations.append(f"🔧 Reduce sulfate from {sulfate_value:.1f} mg/L to <250 mg/L using treatment")
        
        # General water quality recommendations
        recommendations.append("🧪 Regular water testing recommended for quality monitoring")
        recommendations.append("📊 Consider implementing water quality monitoring system")
        
        return recommendations
    
    def _get_changes(self, original: pd.Series, counterfactual: pd.Series) -> Dict:
        """Get changes made in counterfactual"""
        changes = {}
        for feature in original.index:
            if feature in counterfactual:
                if original[feature] != counterfactual[feature]:
                    changes[feature] = {
                        'from': original[feature],
                        'to': counterfactual[feature],
                        'change_type': self._classify_change(feature, original[feature], counterfactual[feature])
                    }
        return changes
    
    def _classify_change(self, feature: str, original: float, new: float) -> str:
        """Classify type of change made"""
        if feature == 'ph':
            if 6.5 <= new <= 8.5:
                return 'OPTIMIZATION'
            else:
                return 'ADJUSTMENT'
        elif feature == 'Turbidity':
            if new < 5.0:
                return 'IMPROVEMENT'
            else:
                return 'ADJUSTMENT'
        else:
            return 'MODIFICATION'
    
    def create_explanation_report(self, explanation: Dict, save_path: str = 'results/xai_report.html'):
        """Create comprehensive XAI report"""
        print("📄 Creating comprehensive XAI report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Water Potability Prediction - XAI Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .violation {{ background-color: #ffebee; }}
                .compliant {{ background-color: #d4edda; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 Water Potability Prediction - XAI Explanation Report</h1>
                <p>Prediction: {'POTABLE' if explanation['prediction'] == 1 else 'NON-POTABLE'}</p>
                <p>Confidence: {explanation['probability']:.4f} ({explanation['probability']*100:.1f}%)</p>
            </div>
            
            <div class="section">
                <h2>🔍 Feature Importance (SHAP Values)</h2>
                <table>
                    <tr><th>Feature</th><th>SHAP Value</th><th>Impact</th></tr>
        """
        
        # Add feature importance table
        for feature, shap_val in explanation['feature_importance']:
            impact = 'POSITIVE' if shap_val > 0 else 'NEGATIVE'
            html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{shap_val:.4f}</td>
                        <td>{impact}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>🔬 Physics Constraints Analysis</h2>
        """
        
        # Add physics constraints
        for param, analysis in explanation['physics_analysis'].items():
            if param != 'overall_compliance':
                status_class = analysis['status'].lower()
                html_content += f"""
                    <div class="metric {status_class}">
                        <strong>{param}</strong>: {analysis['value']} ({analysis['status']})
                        <br>Standard: {analysis.get('limit', analysis.get('range', 'N/A'))}
                    </div>
                """
        
        # Add overall compliance
        overall = explanation['physics_analysis']['overall_compliance']
        html_content += f"""
                <div class="metric {'compliant' if overall['violations'] == 0 else 'violation'}">
                    <strong>Overall Compliance</strong>: {overall['status']} ({overall['score']:.1%})
                    <br>Violations: {overall['violations']}/{len(explanation['physics_analysis'])-1}
                </div>
            </div>
            
            <div class="section">
                <h2>⚡ Counterfactual Explanations</h2>
        """
        
        # Add counterfactuals
        for i, cf in enumerate(explanation['counterfactuals'], 1):
            change_status = "✅ Prediction Changed" if cf['prediction_changed'] else "❌ Prediction Unchanged"
            html_content += f"""
                <div class="metric">
                    <h3>Counterfactual {i}</h3>
                    <p><strong>{change_status}</strong></p>
                    <table>
                        <tr><th>Feature</th><th>Original</th><th>Counterfactual</th><th>Change Type</th></tr>
            """
            
            for feature, change in cf['changes_made'].items():
                html_content += f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{change['from']:.2f}</td>
                            <td>{change['to']:.2f}</td>
                            <td>{change['change_type']}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        html_content += """
            
            <div class="section">
                <h2>🧪 Water Chemistry Analysis</h2>
        """
        
        # Add chemistry insights
        for relationship, analysis in explanation['chemistry_insights'].items():
            html_content += f"""
                <div class="metric">
                    <strong>{relationship.replace('_', ' - ').title()}</strong>: {analysis['status']}
                    <br>{analysis['relationship']}
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>🔧 Recommendations</h2>
        """
        
        # Add recommendations
        for rec in explanation['recommendations']:
            html_content += f"""
                <div class="recommendation">{rec}</div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ XAI report saved to {save_path}")
    
    def explain_batch(self, X_test: pd.DataFrame) -> Dict:
        """Explain multiple predictions"""
        print("🔍 Explaining batch predictions...")
        
        explanations = []
        shap_values = self.explainer.shap_values(X_test)
        
        for i, (idx, row) in enumerate(X_test.iterrows()):
            explanation = self.explain_prediction(row, X_test)
            explanations.append(explanation)
        
        return {
            'shap_values': shap_values,
            'explanations': explanations,
            'summary': self._generate_batch_summary(explanations)
        }
    
    def _generate_batch_summary(self, explanations: List[Dict]) -> Dict:
        """Generate summary statistics for batch explanations"""
        total_predictions = len(explanations)
        potable_predictions = sum(1 for exp in explanations if exp['prediction'] == 1)
        
        avg_confidence = np.mean([exp['probability'] for exp in explanations])
        
        # Most common violations
        violation_counts = {}
        for exp in explanations:
            for param, analysis in exp['physics_analysis'].items():
                if param != 'overall_compliance' and analysis['status'] == 'VIOLATION':
                    violation_counts[param] = violation_counts.get(param, 0) + 1
        
        return {
            'total_predictions': total_predictions,
            'potable_predictions': potable_predictions,
            'non_potable_predictions': total_predictions - potable_predictions,
            'average_confidence': avg_confidence,
            'most_common_violations': violation_counts
        }
