import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class FeatureEngineer:
    """
    Engineers features from processed transaction data for credit scoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def engineer_features(self, df):
        """
        Engineer comprehensive features for credit scoring
        
        Args:
            df: Processed transaction DataFrame
            
        Returns:
            pandas.DataFrame: Feature matrix with wallet addresses as index
        """
        try:
            features_list = []
            
            # Process each wallet
            for wallet in df['userWallet'].unique():
                wallet_data = df[df['userWallet'] == wallet]
                wallet_features = self._extract_wallet_features(wallet_data)
                wallet_features['userWallet'] = wallet
                features_list.append(wallet_features)
            
            # Combine all features
            features_df = pd.DataFrame(features_list)
            features_df.set_index('userWallet', inplace=True)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            # Add interaction features
            features_df = self._add_interaction_features(features_df)
            
            # Scale features
            features_df = self._scale_features(features_df)
            
            self.logger.info(f"Engineered {len(features_df.columns)} features for {len(features_df)} wallets")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            return None
    
    def _extract_wallet_features(self, wallet_data):
        """Extract comprehensive features for a single wallet"""
        features = {}
        
        # Basic transaction features
        features.update(self._basic_transaction_features(wallet_data))
        
        # Volume and value features
        features.update(self._volume_value_features(wallet_data))
        
        # Behavioral features
        features.update(self._behavioral_features(wallet_data))
        
        # Risk features
        features.update(self._risk_features(wallet_data))
        
        # Diversification features
        features.update(self._diversification_features(wallet_data))
        
        # Time-based features
        features.update(self._time_based_features(wallet_data))
        
        # Consistency features
        features.update(self._consistency_features(wallet_data))
        
        return features
    
    def _basic_transaction_features(self, data):
        """Basic transaction count and frequency features"""
        features = {}
        
        # Transaction counts
        features['total_transactions'] = len(data)
        features['unique_tx_hashes'] = data['txHash'].nunique()
        
        # Action distribution
        action_counts = data['action'].value_counts()
        for action in ['deposit', 'redeemunderlying', 'borrow', 'repay', 'liquidationcall']:
            features[f'{action}_count'] = action_counts.get(action, 0)
            features[f'{action}_ratio'] = action_counts.get(action, 0) / len(data)
        
        # Time span
        time_span = (data['datetime'].max() - data['datetime'].min()).days
        features['activity_span_days'] = max(time_span, 1)
        features['transaction_frequency'] = len(data) / max(time_span, 1)
        
        return features
    
    def _volume_value_features(self, data):
        """Volume and value-based features"""
        features = {}
        
        # USD value statistics
        usd_values = data['normalized_usd_value'].dropna()
        if len(usd_values) > 0:
            features['total_volume_usd'] = usd_values.sum()
            features['avg_transaction_usd'] = usd_values.mean()
            features['median_transaction_usd'] = usd_values.median()
            features['max_transaction_usd'] = usd_values.max()
            features['min_transaction_usd'] = usd_values.min()
            features['std_transaction_usd'] = usd_values.std()
            features['volume_coefficient_variation'] = usd_values.std() / usd_values.mean() if usd_values.mean() > 0 else 0
        else:
            for key in ['total_volume_usd', 'avg_transaction_usd', 'median_transaction_usd', 
                       'max_transaction_usd', 'min_transaction_usd', 'std_transaction_usd', 
                       'volume_coefficient_variation']:
                features[key] = 0
        
        # Size category distribution
        size_dist = data['size_category'].value_counts()
        for size in ['micro', 'small', 'medium', 'large', 'whale']:
            features[f'{size}_transactions'] = size_dist.get(size, 0)
            features[f'{size}_ratio'] = size_dist.get(size, 0) / len(data)
        
        return features
    
    def _behavioral_features(self, data):
        """Behavioral pattern features"""
        features = {}
        
        # Deposit/withdrawal behavior
        deposits = data[data['action'] == 'deposit']
        withdrawals = data[data['action'] == 'redeemunderlying']
        
        deposit_volume = deposits['normalized_usd_value'].sum()
        withdrawal_volume = withdrawals['normalized_usd_value'].sum()
        
        features['deposit_volume'] = deposit_volume
        features['withdrawal_volume'] = withdrawal_volume
        features['net_deposit_volume'] = deposit_volume - withdrawal_volume
        features['deposit_withdrawal_ratio'] = deposit_volume / max(withdrawal_volume, 1)
        
        # Borrowing behavior
        borrows = data[data['action'] == 'borrow']
        repays = data[data['action'] == 'repay']
        
        borrow_volume = borrows['normalized_usd_value'].sum()
        repay_volume = repays['normalized_usd_value'].sum()
        
        features['borrow_volume'] = borrow_volume
        features['repay_volume'] = repay_volume
        features['net_borrow_volume'] = borrow_volume - repay_volume
        features['borrow_repay_ratio'] = borrow_volume / max(repay_volume, 1)
        
        # Transaction timing patterns
        features['avg_time_between_transactions'] = self._calculate_avg_time_between(data)
        features['transaction_regularity'] = self._calculate_regularity(data)
        
        return features
    
    def _risk_features(self, data):
        """Risk-related features"""
        features = {}
        
        # Liquidation events
        liquidations = data[data['action'] == 'liquidationcall']
        features['liquidation_count'] = len(liquidations)
        features['liquidation_volume'] = liquidations['normalized_usd_value'].sum()
        features['liquidation_ratio'] = len(liquidations) / len(data)
        
        # Risk indicators
        features['has_liquidations'] = int(len(liquidations) > 0)
        features['high_value_transactions'] = int(data['normalized_usd_value'].max() > 100000)
        features['suspicious_frequency'] = int(len(data) / max(features.get('activity_span_days', 1), 1) > 100)
        
        # Borrowing risk
        if features.get('borrow_volume', 0) > 0:
            features['leverage_ratio'] = features['borrow_volume'] / max(features.get('deposit_volume', 1), 1)
            features['repayment_completeness'] = features.get('repay_volume', 0) / features['borrow_volume']
        else:
            features['leverage_ratio'] = 0
            features['repayment_completeness'] = 1
        
        return features
    
    def _diversification_features(self, data):
        """Portfolio diversification features"""
        features = {}
        
        # Asset diversification
        features['unique_assets'] = data['assetSymbol'].nunique()
        features['asset_concentration'] = self._calculate_concentration(data, 'assetSymbol')
        
        # Network diversification
        features['unique_networks'] = data['network'].nunique()
        features['network_concentration'] = self._calculate_concentration(data, 'network')
        
        # Protocol diversification
        features['unique_protocols'] = data['protocol'].nunique()
        features['protocol_concentration'] = self._calculate_concentration(data, 'protocol')
        
        # Pool diversification
        features['unique_pools'] = data['poolId'].nunique()
        features['pool_concentration'] = self._calculate_concentration(data, 'poolId')
        
        return features
    
    def _time_based_features(self, data):
        """Time-based activity features"""
        features = {}
        
        # Hour of day patterns
        hour_dist = data['hour'].value_counts()
        features['most_active_hour'] = hour_dist.index[0] if len(hour_dist) > 0 else 0
        features['hour_diversity'] = len(hour_dist)
        features['night_activity_ratio'] = len(data[data['hour'].isin([0, 1, 2, 3, 4, 5])]) / len(data)
        
        # Day of week patterns
        dow_dist = data['day_of_week'].value_counts()
        features['most_active_dow'] = dow_dist.index[0] if len(dow_dist) > 0 else 0
        features['dow_diversity'] = len(dow_dist)
        features['weekend_activity_ratio'] = len(data[data['day_of_week'].isin([5, 6])]) / len(data)
        
        # Activity consistency over time
        features['activity_consistency'] = self._calculate_activity_consistency(data)
        
        return features
    
    def _consistency_features(self, data):
        """Consistency and stability features"""
        features = {}
        
        # Transaction amount consistency
        amounts = data['normalized_usd_value'].dropna()
        if len(amounts) > 1:
            features['amount_stability'] = 1 / (1 + amounts.std() / amounts.mean()) if amounts.mean() > 0 else 0
        else:
            features['amount_stability'] = 1
        
        # Action consistency
        action_entropy = self._calculate_entropy(data['action'].value_counts())
        features['action_entropy'] = action_entropy
        features['action_consistency'] = 1 / (1 + action_entropy)
        
        # Time consistency
        features['time_consistency'] = self._calculate_time_consistency(data)
        
        return features
    
    def _calculate_avg_time_between(self, data):
        """Calculate average time between transactions"""
        if len(data) < 2:
            return 0
        
        sorted_data = data.sort_values('timestamp')
        time_diffs = sorted_data['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return 0
        
        return time_diffs.mean() / 3600  # Convert to hours
    
    def _calculate_regularity(self, data):
        """Calculate transaction regularity score"""
        if len(data) < 3:
            return 0
        
        sorted_data = data.sort_values('timestamp')
        time_diffs = sorted_data['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return 0
        
        regularity = 1 / (1 + time_diffs.std() / time_diffs.mean()) if time_diffs.mean() > 0 else 0
        return regularity
    
    def _calculate_concentration(self, data, column):
        """Calculate concentration index (Herfindahl-Hirschman Index)"""
        if column not in data.columns:
            return 0
        
        counts = data[column].value_counts()
        proportions = counts / len(data)
        hhi = (proportions ** 2).sum()
        
        return hhi
    
    def _calculate_activity_consistency(self, data):
        """Calculate activity consistency over time"""
        if len(data) < 7:
            return 0
        
        # Group by day and count transactions
        data_copy = data.copy()
        data_copy['date'] = data_copy['datetime'].dt.date
        daily_counts = data_copy.groupby('date').size()
        
        if len(daily_counts) < 2:
            return 0
        
        # Calculate consistency as inverse of coefficient of variation
        consistency = 1 / (1 + daily_counts.std() / daily_counts.mean()) if daily_counts.mean() > 0 else 0
        return consistency
    
    def _calculate_entropy(self, counts):
        """Calculate entropy of a distribution"""
        if len(counts) == 0:
            return 0
        
        proportions = counts / counts.sum()
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        
        return entropy
    
    def _calculate_time_consistency(self, data):
        """Calculate consistency of transaction timing"""
        if len(data) < 5:
            return 0
        
        # Group by hour and calculate consistency
        hourly_counts = data.groupby('hour').size()
        
        if len(hourly_counts) < 2:
            return 0
        
        consistency = 1 / (1 + hourly_counts.std() / hourly_counts.mean()) if hourly_counts.mean() > 0 else 0
        return consistency
    
    def _handle_missing_values(self, df):
        """Handle missing values in feature matrix"""
        # Fill NaN values with 0 for most features
        df = df.fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _add_interaction_features(self, df):
        """Add interaction features between key variables"""
        # Volume-frequency interactions
        df['volume_per_transaction'] = df['total_volume_usd'] / (df['total_transactions'] + 1)
        df['volume_consistency_score'] = df['volume_coefficient_variation'] * df['activity_consistency']
        
        # Risk-behavior interactions
        df['risk_adjusted_volume'] = df['total_volume_usd'] * (1 - df['liquidation_ratio'])
        df['leverage_activity_score'] = df['leverage_ratio'] * df['transaction_frequency']
        
        # Diversification-stability interactions
        df['diversification_stability'] = df['asset_concentration'] * df['amount_stability']
        
        return df
    
    def _scale_features(self, df):
        """Scale features to appropriate ranges"""
        # Log transform highly skewed features
        log_features = ['total_volume_usd', 'avg_transaction_usd', 'max_transaction_usd']
        
        for feature in log_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(df[feature])
        
        # Clip extreme values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(q01, q99)
        
        return df
