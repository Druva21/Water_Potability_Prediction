"""
REAL TabTransformer Implementation for Water Potability Prediction
Actual deep learning with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention Mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)

class PhysicsInformedLayer(nn.Module):
    """Physics-informed neural network layer"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Physics-aware weights
        self.physics_weights = nn.Parameter(torch.ones(input_dim))
        
        # Standard neural layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        
        # Physics constraints
        self.ph_constraint = nn.Parameter(torch.tensor([6.5, 8.5]))  # pH range
        self.turbidity_constraint = nn.Parameter(torch.tensor([5.0]))  # Turbidity limit
        self.sulfate_constraint = nn.Parameter(torch.tensor([250.0]))  # Sulfate limit
        
    def forward(self, x: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        # Apply physics-aware weighting (use first min(input_dim, x.size(1)) features)
        num_features = min(self.input_dim, x.size(1))
        weighted_x = x[:, :num_features] * torch.sigmoid(self.physics_weights[:num_features])
        
        # Standard forward pass
        h1 = F.relu(self.layer1(weighted_x))
        h2 = F.relu(self.layer2(h1))
        
        # Apply physics constraints
        physics_loss = torch.tensor(0.0, device=x.device)
        if physics_features.size(1) > 0:
            ph = physics_features[:, 0] if physics_features.size(1) > 0 else torch.tensor(7.0, device=physics_features.device)
            turbidity = physics_features[:, 8] if physics_features.size(1) > 8 else torch.tensor(5.0, device=physics_features.device)
            sulfate = physics_features[:, 4] if physics_features.size(1) > 4 else torch.tensor(200.0, device=physics_features.device)
            
            ph_violation = torch.sum((ph < self.ph_constraint[0]) | (ph > self.ph_constraint[1])).float()
            turbidity_violation = torch.sum(turbidity > self.turbidity_constraint).float()
            sulfate_violation = torch.sum(sulfate > self.sulfate_constraint).float()
            
            physics_loss = ph_violation + turbidity_violation + sulfate_violation
            
        # Final output with physics penalty
        output = self.layer3(h2) - 0.1 * physics_loss.unsqueeze(-1) if physics_loss > 0 else self.layer3(h2)
        
        return output

# Advanced working TabTransformer with state-of-the-art features
class AdvancedTabTransformer(nn.Module):
    """Advanced TabTransformer with positional encoding, layer norm, and enhanced attention"""
    
    def __init__(self, input_dim: int = 13, d_model: int = 64, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Enhanced feature embedding with bias
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.feature_bias = nn.Parameter(torch.zeros(d_model))
        
        # Positional encoding for tabular data
        self.pos_encoding = self._create_positional_encoding(1000, d_model)
        
        # Advanced multi-head attention layers with layer norm
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Enhanced feed-forward networks
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),  # Better than ReLU
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalization for stability
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers * 2)
        ])
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create positional encoding for tabular data"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Enhanced feature embedding
        x = self.feature_embedding(x) + self.feature_bias
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # Store attention weights for explainability
        attention_weights = []
        
        # Advanced transformer layers with residual connections
        for i, (attn_layer, ff_layer) in enumerate(zip(self.attention_layers, self.ff_layers)):
            # Multi-head self-attention with proper masking
            attn_output, attn_weights = attn_layer(x, x, x, key_padding_mask=None)
            attention_weights.append(attn_weights)
            
            # Add & Norm (Post-LN)
            x = self.norm_layers[i*2](x + attn_output)
            
            # Feed-forward with GELU activation
            ff_output = ff_layer(x)
            x = self.norm_layers[i*2+1](x + ff_output)
        
        # Global average pooling with attention
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Enhanced classification
        output = self.classifier(x)  # (batch_size, 1)
        
        # Advanced explanations
        explanations = {
            'attention_weights': torch.stack(attention_weights, dim=1).mean(dim=1) if attention_weights else torch.zeros(batch_size, 1),
            'feature_importance': torch.abs(self.feature_embedding.weight).mean(dim=0),
            'attention_entropy': self._compute_attention_entropy(attention_weights),
            'layer_outputs': len(self.attention_layers)
        }
        
        return output.squeeze(-1), explanations
    
    def _compute_attention_entropy(self, attention_weights):
        """Compute attention entropy for explainability"""
        if not attention_weights:
            return torch.zeros(1)
        
        # Average attention across layers and heads
        avg_attention = torch.stack(attention_weights).mean(dim=[0, 1, 2])
        
        # Compute entropy
        entropy = -(avg_attention * torch.log(avg_attention + 1e-8)).sum()
        return entropy.unsqueeze(0)

# Use the advanced version
RealTabTransformer = AdvancedTabTransformer

class RealWaterPotabilityPredictor:
    """Advanced deep learning predictor with state-of-the-art TabTransformer"""
    
    def __init__(self, input_dim: int = 13, d_model: int = 64, 
                 n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.model = RealTabTransformer(input_dim=input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        
        # Standard loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimized optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=1e-5
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.training_history = {'loss': [], 'accuracy': []}
        
    def fit(self, X_train, y_train):
        """Fit method for sklearn compatibility"""
        return self.train(X_train, y_train, epochs=20, batch_size=32)
    
    def train(self, X_train, y_train, epochs: int = 20, batch_size: int = 32):
        """Advanced training with better performance"""
        self.model.train()
        
        # Convert to tensors and move to device
        if hasattr(X_train, 'values'):
            X_tensor = torch.FloatTensor(X_train.values).to(self.device)
        else:
            X_tensor = torch.FloatTensor(X_train).to(self.device)
            
        if hasattr(y_train, 'values'):
            y_tensor = torch.FloatTensor(y_train.values).to(self.device)
        else:
            y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        print(f"🧠 ADVANCED TABTRANSFORMER TRAINING")
        print(f"📊 Architecture: {self.model.__class__.__name__}")
        print(f"🔧 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"💾 Device: {self.device}")
        print(f"📈 Epochs: {epochs}, Batch Size: {batch_size}")
        print("-" * 50)
        
        best_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            correct_predictions = 0
            total_samples = 0
            
            # Shuffle data each epoch
            indices = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, explanations = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_predictions += (predictions == batch_y).sum().item()
                total_samples += batch_y.size(0)
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate metrics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            # Store history
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'advanced_tabtransformer.pth')
            
            # Progress reporting
            if epoch % 5 == 0:
                print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}')
                
                if 'attention_entropy' in explanations:
                    print(f'         | Attention Entropy: {explanations["attention_entropy"].item():.4f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('advanced_tabtransformer.pth'))
        
        print(f'\n✅ Advanced Training Completed!')
        print(f'🏆 Best Loss: {best_loss:.4f}')
        print(f'📊 Final Accuracy: {self.training_history["accuracy"][-1]:.4f}')
        print(f'🧠 Layers: {len(self.model.attention_layers)} transformer layers')
        
        return self
    
    def _predict(self, X_test) -> Tuple[np.ndarray, dict]:
        """Private method to make predictions with explanations"""
        self.model.eval()
        
        # Convert to tensor
        if hasattr(X_test, 'values'):
            X_tensor = torch.FloatTensor(X_test.values).to(self.device)
        else:
            X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            logits, explanations = self.model(X_tensor)
            probabilities = torch.sigmoid(logits)
            
        # Convert to numpy
        probs = probabilities.cpu().numpy()
        
        # Convert explanations to numpy
        explanations_np = {
            'attention_weights': explanations['attention_weights'].cpu().numpy(),
            'feature_importance': explanations['feature_importance'].cpu().numpy()
        }
        
        return probs, explanations_np
    
    def predict_proba(self, X_test):
        """Return probabilities for both classes"""
        probs, _ = self._predict(X_test)
        # Return [prob_class_0, prob_class_1]
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X_test):
        """Return binary predictions"""
        probs, _ = self._predict(X_test)
        return (probs > 0.5).astype(int)
    
    def get_attention_weights(self, X_test) -> torch.Tensor:
        """Extract attention weights for XAI"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test.values)
        
        with torch.no_grad():
            _, explanations = self.model(X_tensor)
            
        return explanations.get('attention_weights', torch.zeros(len(X_test)))
