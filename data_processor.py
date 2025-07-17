import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

class DataProcessor:
    """
    Processes raw Aave V2 transaction data for credit scoring analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_transactions(self, raw_data):
        """
        Process raw transaction data into structured format
        
        Args:
            raw_data: List of transaction dictionaries
            
        Returns:
            pandas.DataFrame: Processed transaction data
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            if len(df) == 0:
                self.logger.warning("No transaction data found")
                return None
            
            # Clean and standardize data
            df = self._clean_data(df)
            
            # Extract action data
            df = self._extract_action_data(df)
            
            # Add derived fields
            df = self._add_derived_fields(df)
            
            # Validate data quality
            df = self._validate_data(df)
            
            self.logger.info(f"Processed {len(df)} transactions for {df['userWallet'].nunique()} wallets")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing transactions: {str(e)}")
            return None
    
    def _clean_data(self, df):
        """Clean and standardize transaction data"""
        # Remove duplicates based on transaction hash and log ID
        df = df.drop_duplicates(subset=['txHash', 'logId'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        
        # Standardize wallet addresses
        df['userWallet'] = df['userWallet'].str.lower()
        
        # Clean action names
        df['action'] = df['action'].str.lower().str.strip()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['userWallet', 'timestamp', 'action'])
        
        return df
    
    def _extract_action_data(self, df):
        """Extract and flatten action data"""
        # Extract amount and asset information
        df['amount'] = df['actionData'].apply(lambda x: self._safe_extract(x, 'amount'))
        df['assetSymbol'] = df['actionData'].apply(lambda x: self._safe_extract(x, 'assetSymbol'))
        df['assetPriceUSD'] = df['actionData'].apply(lambda x: self._safe_extract(x, 'assetPriceUSD'))
        df['poolId'] = df['actionData'].apply(lambda x: self._safe_extract(x, 'poolId'))
        
        # Convert numeric fields
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce')
        
        # Calculate USD value
        df['usd_value'] = df['amount'] * df['assetPriceUSD']
        
        # Handle different asset decimal places (common DeFi tokens)
        df['normalized_amount'] = df.apply(self._normalize_amount, axis=1)
        df['normalized_usd_value'] = df['normalized_amount'] * df['assetPriceUSD']
        
        return df
    
    def _safe_extract(self, action_data, key):
        """Safely extract value from action data"""
        if isinstance(action_data, dict):
            return action_data.get(key)
        return None
    
    def _normalize_amount(self, row):
        """Normalize amounts based on token decimals"""
        amount = row['amount']
        asset = row['assetSymbol']
        
        if pd.isna(amount) or pd.isna(asset):
            return 0
        
        # Common token decimals
        decimals_map = {
            'USDC': 6,
            'USDT': 6,
            'DAI': 18,
            'WETH': 18,
            'WMATIC': 18,
            'WBTC': 8,
            'AAVE': 18
        }
        
        decimals = decimals_map.get(asset, 18)
        return amount / (10 ** decimals)
    
    def _add_derived_fields(self, df):
        """Add derived fields for analysis"""
        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # Transaction size categories
        df['size_category'] = pd.cut(
            df['normalized_usd_value'],
            bins=[0, 100, 1000, 10000, 100000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'whale']
        )
        
        # Action categories
        df['action_category'] = df['action'].map({
            'deposit': 'supply',
            'redeemunderlying': 'withdraw',
            'borrow': 'borrow',
            'repay': 'repay',
            'liquidationcall': 'liquidation'
        })
        
        return df
    
    def _validate_data(self, df):
        """Validate data quality and remove invalid records"""
        # Remove transactions with invalid amounts
        df = df[df['normalized_usd_value'] > 0]
        
        # Remove transactions with invalid timestamps
        df = df[df['timestamp'] > 0]
        
        # Remove wallets with suspicious patterns (optional)
        # This could be expanded with more sophisticated filtering
        
        return df
    
    def get_wallet_summary(self, df, wallet_address):
        """
        Get summary statistics for a specific wallet
        
        Args:
            df: Processed transaction DataFrame
            wallet_address: Wallet address to analyze
            
        Returns:
            dict: Wallet summary statistics
        """
        wallet_data = df[df['userWallet'] == wallet_address.lower()]
        
        if len(wallet_data) == 0:
            return None
        
        summary = {
            'wallet_address': wallet_address,
            'total_transactions': len(wallet_data),
            'first_transaction': wallet_data['datetime'].min(),
            'last_transaction': wallet_data['datetime'].max(),
            'total_volume_usd': wallet_data['normalized_usd_value'].sum(),
            'average_transaction_usd': wallet_data['normalized_usd_value'].mean(),
            'unique_assets': wallet_data['assetSymbol'].nunique(),
            'action_distribution': wallet_data['action'].value_counts().to_dict(),
            'networks': wallet_data['network'].unique().tolist(),
            'protocols': wallet_data['protocol'].unique().tolist()
        }
        
        return summary
    
    def get_dataset_summary(self, df):
        """
        Get summary statistics for the entire dataset
        
        Args:
            df: Processed transaction DataFrame
            
        Returns:
            dict: Dataset summary statistics
        """
        return {
            'total_transactions': len(df),
            'unique_wallets': df['userWallet'].nunique(),
            'date_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max()
            },
            'total_volume_usd': df['normalized_usd_value'].sum(),
            'action_distribution': df['action'].value_counts().to_dict(),
            'asset_distribution': df['assetSymbol'].value_counts().to_dict(),
            'network_distribution': df['network'].value_counts().to_dict(),
            'protocol_distribution': df['protocol'].value_counts().to_dict()
        }
