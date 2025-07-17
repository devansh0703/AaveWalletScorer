import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging

class CreditScoringModel:
    """
    Machine learning model for DeFi credit scoring
    """
    
    def __init__(self, n_estimators=100, test_size=0.2, random_state=42):
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)
        
    def train_and_score(self, features_df):
        """
        Train model and generate credit scores
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            tuple: (scores_df, feature_importance, model_metrics)
        """
        try:
            # Create target variable (unsupervised scoring)
            target = self._create_target_variable(features_df)
            
            # Prepare features
            X = features_df.copy()
            X = self._prepare_features(X)
            
            # Train model
            model_metrics = self._train_model(X, target)
            
            # Generate scores
            scores = self._generate_scores(X)
            
            # Create results DataFrame
            scores_df = pd.DataFrame({
                'userWallet': features_df.index,
                'credit_score': scores
            })
            
            # Get feature importance
            feature_importance = self._get_feature_importance(X.columns)
            
            return scores_df, feature_importance, model_metrics
            
        except Exception as e:
            self.logger.error(f"Error in train_and_score: {str(e)}")
            return None
    
    def _create_target_variable(self, features_df):
        """
        Create target variable for supervised learning using composite scoring
        """
        # Normalize features for composite scoring
        df_norm = features_df.copy()
        
        # Key positive indicators (higher is better)
        positive_features = [
            'total_volume_usd', 'deposit_volume', 'repay_volume',
            'unique_assets', 'activity_consistency', 'amount_stability',
            'repayment_completeness', 'transaction_regularity'
        ]
        
        # Key negative indicators (lower is better)
        negative_features = [
            'liquidation_count', 'liquidation_ratio', 'leverage_ratio',
            'suspicious_frequency', 'night_activity_ratio'
        ]
        
        # Normalize and weight features
        target_score = np.zeros(len(df_norm))
        
        # Positive contributions
        for feature in positive_features:
            if feature in df_norm.columns:
                normalized = self._normalize_feature(df_norm[feature])
                weight = self._get_feature_weight(feature)
                target_score += normalized * weight
        
        # Negative contributions
        for feature in negative_features:
            if feature in df_norm.columns:
                normalized = self._normalize_feature(df_norm[feature])
                weight = self._get_feature_weight(feature)
                target_score -= normalized * weight
        
        # Scale to reasonable range
        target_score = (target_score - target_score.min()) / (target_score.max() - target_score.min())
        target_score = target_score * 100  # Scale to 0-100 for training
        
        return target_score
    
    def _normalize_feature(self, series):
        """Normalize feature to 0-1 range"""
        if series.std() == 0:
            return np.zeros(len(series))
        return (series - series.min()) / (series.max() - series.min())
    
    def _get_feature_weight(self, feature):
        """Get weight for feature in composite scoring"""
        weights = {
            # Positive features
            'total_volume_usd': 0.2,
            'deposit_volume': 0.15,
            'repay_volume': 0.15,
            'unique_assets': 0.1,
            'activity_consistency': 0.1,
            'amount_stability': 0.1,
            'repayment_completeness': 0.2,
            'transaction_regularity': 0.05,
            
            # Negative features
            'liquidation_count': 0.3,
            'liquidation_ratio': 0.25,
            'leverage_ratio': 0.15,
            'suspicious_frequency': 0.2,
            'night_activity_ratio': 0.1
        }
        
        return weights.get(feature, 0.05)
    
    def _prepare_features(self, X):
        """Prepare features for model training"""
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Fill missing values
        X = X.fillna(0)
        
        # Remove constant columns
        constant_columns = X.columns[X.var() == 0]
        X = X.drop(columns=constant_columns)
        
        return X
    
    def _train_model(self, X, y):
        """Train the Random Forest model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='r2'
        )
        
        metrics = {
            'test_r2': r2_score(y_test, y_pred),
            'test_mse': mean_squared_error(y_test, y_pred),
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.logger.info(f"Model trained with R² = {metrics['test_r2']:.3f}")
        
        return metrics
    
    def _generate_scores(self, X):
        """Generate credit scores for all wallets"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict scores
        raw_scores = self.model.predict(X_scaled)
        
        # Convert to 0-1000 scale
        scores = self._scale_to_credit_range(raw_scores)
        
        # Apply business rules
        scores = self._apply_business_rules(scores, X)
        
        return scores
    
    def _scale_to_credit_range(self, raw_scores):
        """Scale raw scores to 0-1000 credit score range"""
        # Normalize to 0-1 range
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score == min_score:
            normalized = np.full(len(raw_scores), 0.5)
        else:
            normalized = (raw_scores - min_score) / (max_score - min_score)
        
        # Scale to 0-1000 range
        credit_scores = normalized * 1000
        
        # Ensure scores are within bounds
        credit_scores = np.clip(credit_scores, 0, 1000)
        
        return credit_scores
    
    def _apply_business_rules(self, scores, X):
        """Apply business rules to adjust scores"""
        adjusted_scores = scores.copy()
        
        # Severe penalties for high-risk behavior
        if 'liquidation_count' in X.columns:
            liquidation_penalty = X['liquidation_count'] * 100
            adjusted_scores = adjusted_scores - liquidation_penalty
        
        if 'suspicious_frequency' in X.columns:
            bot_penalty = X['suspicious_frequency'] * 200
            adjusted_scores = adjusted_scores - bot_penalty
        
        # Bonuses for good behavior
        if 'repayment_completeness' in X.columns:
            repayment_bonus = (X['repayment_completeness'] - 0.5) * 100
            adjusted_scores = adjusted_scores + repayment_bonus
        
        if 'activity_consistency' in X.columns:
            consistency_bonus = X['activity_consistency'] * 50
            adjusted_scores = adjusted_scores + consistency_bonus
        
        # Ensure scores remain in valid range
        adjusted_scores = np.clip(adjusted_scores, 0, 1000)
        
        return adjusted_scores
    
    def _get_feature_importance(self, feature_names):
        """Get feature importance from trained model"""
        if self.model is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_single_wallet(self, wallet_features):
        """
        Predict credit score for a single wallet
        
        Args:
            wallet_features: Series or dict with wallet features
            
        Returns:
            float: Credit score (0-1000)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_and_score first.")
        
        # Convert to DataFrame if needed
        if isinstance(wallet_features, dict):
            wallet_features = pd.Series(wallet_features)
        
        # Prepare features
        X = wallet_features.values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        raw_score = self.model.predict(X_scaled)[0]
        
        # Scale to credit range
        credit_score = self._scale_to_credit_range(np.array([raw_score]))[0]
        
        return credit_score
    
    def get_score_explanation(self, wallet_features, top_n=10):
        """
        Get explanation for a wallet's credit score
        
        Args:
            wallet_features: Series or dict with wallet features
            top_n: Number of top features to explain
            
        Returns:
            dict: Explanation with feature contributions
        """
        if self.model is None or self.feature_importance is None:
            return None
        
        # Get top features
        top_features = self.feature_importance.head(top_n)
        
        explanation = {
            'top_features': top_features.to_dict('records'),
            'wallet_values': {}
        }
        
        # Get wallet values for top features
        for feature in top_features['feature']:
            if feature in wallet_features.index:
                explanation['wallet_values'][feature] = wallet_features[feature]
        
        return explanation
