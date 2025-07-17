import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from datetime import datetime, timedelta

from data_processor import DataProcessor
from feature_engineering import FeatureEngineer
from credit_scoring_model import CreditScoringModel

# Page configuration
st.set_page_config(
    page_title="DeFi Credit Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏦 DeFi Credit Scoring System")
st.markdown("""
This system analyzes Aave V2 transaction patterns to assign wallet creditworthiness scores from 0-1000.
Higher scores indicate reliable and responsible usage; lower scores reflect risky, bot-like, or exploitative behavior.
""")

# Sidebar for file upload and controls
st.sidebar.header("Data Input")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Transaction Data (JSON)",
    type=['json'],
    help="Upload JSON file containing Aave V2 transaction data"
)

# Load sample data if available
sample_data_path = "attached_assets/user-wallet-transactions_1752780140063.json"
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True if os.path.exists(sample_data_path) else False)

# Model parameters
st.sidebar.header("Model Parameters")
min_transactions = st.sidebar.slider("Minimum Transactions per Wallet", 1, 20, 5)
n_estimators = st.sidebar.slider("Random Forest Estimators", 50, 500, 100)
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize data processor, feature engineer, and model"""
    processor = DataProcessor()
    engineer = FeatureEngineer()
    model = CreditScoringModel(n_estimators=n_estimators, test_size=test_size)
    return processor, engineer, model

processor, engineer, model = initialize_components()

# Load and process data
@st.cache_data
def load_and_process_data(file_path=None, file_content=None):
    """Load and process transaction data"""
    try:
        if file_content:
            # Load from uploaded file
            data = json.loads(file_content.decode('utf-8'))
        elif file_path and os.path.exists(file_path):
            # Load from sample file
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            return None, "No data source available"
        
        # Process the data
        df = processor.process_transactions(data)
        
        if df is None or len(df) == 0:
            return None, "No valid transactions found"
        
        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# Main application logic
if uploaded_file or (use_sample_data and os.path.exists(sample_data_path)):
    # Load data
    if uploaded_file:
        df, error = load_and_process_data(file_content=uploaded_file.read())
    else:
        df, error = load_and_process_data(file_path=sample_data_path)
    
    if error:
        st.error(error)
    else:
        # Data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Unique Wallets", df['userWallet'].nunique())
        with col3:
            st.metric("Protocols", df['protocol'].nunique())
        with col4:
            st.metric("Networks", df['network'].nunique())
        
        # Transaction distribution
        st.subheader("Transaction Distribution")
        fig_dist = px.histogram(df, x='action', title='Transaction Types Distribution')
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature engineering
        st.header("🔧 Feature Engineering")
        
        with st.spinner("Engineering features..."):
            # Filter wallets with minimum transactions
            wallet_counts = df['userWallet'].value_counts()
            eligible_wallets = wallet_counts[wallet_counts >= min_transactions].index
            filtered_df = df[df['userWallet'].isin(eligible_wallets)]
            
            if len(filtered_df) == 0:
                st.error(f"No wallets found with at least {min_transactions} transactions")
            else:
                # Engineer features
                features_df = engineer.engineer_features(filtered_df)
                
                st.success(f"Engineered {len(features_df.columns)} features for {len(features_df)} wallets")
                
                # Show feature statistics
                st.subheader("Feature Statistics")
                feature_stats = features_df.describe()
                st.dataframe(feature_stats)
                
                # Feature importance analysis
                st.header("🤖 Model Training & Scoring")
                
                with st.spinner("Training model and generating scores..."):
                    # Train model and generate scores
                    results = model.train_and_score(features_df)
                    
                    if results:
                        scores_df, feature_importance, model_metrics = results
                        
                        # Display model metrics
                        st.subheader("Model Performance")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Cross-validation Score", f"{model_metrics['cv_score']:.3f}")
                        with col2:
                            st.metric("Feature Count", len(feature_importance))
                        with col3:
                            st.metric("Scored Wallets", len(scores_df))
                        
                        # Feature importance plot
                        st.subheader("Feature Importance")
                        fig_importance = px.bar(
                            feature_importance.head(15),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 15 Most Important Features'
                        )
                        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Credit score distribution
                        st.header("📈 Credit Score Analysis")
                        
                        # Score distribution
                        st.subheader("Score Distribution")
                        fig_scores = px.histogram(
                            scores_df,
                            x='credit_score',
                            nbins=50,
                            title='Credit Score Distribution'
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)
                        
                        # Score range analysis
                        st.subheader("Score Range Analysis")
                        
                        # Create score ranges
                        score_ranges = [
                            (0, 100, "Very Poor (0-100)"),
                            (100, 200, "Poor (100-200)"),
                            (200, 400, "Fair (200-400)"),
                            (400, 600, "Good (400-600)"),
                            (600, 800, "Very Good (600-800)"),
                            (800, 1000, "Excellent (800-1000)")
                        ]
                        
                        range_data = []
                        for min_score, max_score, label in score_ranges:
                            count = len(scores_df[(scores_df['credit_score'] >= min_score) & 
                                                (scores_df['credit_score'] < max_score)])
                            percentage = (count / len(scores_df)) * 100
                            range_data.append({
                                'Range': label,
                                'Count': count,
                                'Percentage': percentage
                            })
                        
                        range_df = pd.DataFrame(range_data)
                        
                        # Display range distribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(range_df.style.format({'Percentage': '{:.1f}%'}))
                        
                        with col2:
                            fig_range = px.bar(
                                range_df,
                                x='Range',
                                y='Count',
                                title='Wallet Distribution by Score Range'
                            )
                            fig_range.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_range, use_container_width=True)
                        
                        # Behavioral analysis by score range
                        st.subheader("Behavioral Analysis by Score Range")
                        
                        # Analyze behavior patterns for each range
                        def analyze_score_range(min_score, max_score, range_name):
                            range_wallets = scores_df[(scores_df['credit_score'] >= min_score) & 
                                                    (scores_df['credit_score'] < max_score)]
                            
                            if len(range_wallets) == 0:
                                return None
                            
                            # Get transaction data for these wallets
                            range_transactions = filtered_df[filtered_df['userWallet'].isin(range_wallets['userWallet'])]
                            
                            # Calculate behavioral metrics
                            behavior_metrics = {
                                'avg_transactions_per_wallet': len(range_transactions) / len(range_wallets),
                                'avg_volume_per_wallet': range_transactions['normalized_usd_value'].sum() / len(range_wallets),
                                'liquidation_rate': (range_transactions['action'] == 'liquidationcall').sum() / len(range_transactions),
                                'deposit_ratio': (range_transactions['action'] == 'deposit').sum() / len(range_transactions),
                                'withdrawal_ratio': (range_transactions['action'] == 'redeemunderlying').sum() / len(range_transactions),
                                'borrow_ratio': (range_transactions['action'] == 'borrow').sum() / len(range_transactions),
                                'repay_ratio': (range_transactions['action'] == 'repay').sum() / len(range_transactions),
                                'avg_asset_diversity': range_transactions.groupby('userWallet')['assetSymbol'].nunique().mean(),
                                'night_activity_ratio': len(range_transactions[pd.to_datetime(range_transactions['timestamp'], unit='s').dt.hour.isin([0,1,2,3,4,5])]) / len(range_transactions)
                            }
                            
                            return behavior_metrics
                        
                        # Create behavioral comparison
                        behavioral_data = []
                        for min_score, max_score, label in score_ranges:
                            metrics = analyze_score_range(min_score, max_score, label)
                            if metrics:
                                behavioral_data.append({
                                    'Range': label,
                                    'Avg Transactions': f"{metrics['avg_transactions_per_wallet']:.1f}",
                                    'Avg Volume ($)': f"{metrics['avg_volume_per_wallet']:.0f}",
                                    'Liquidation Rate': f"{metrics['liquidation_rate']:.3f}",
                                    'Deposit Ratio': f"{metrics['deposit_ratio']:.3f}",
                                    'Borrow Ratio': f"{metrics['borrow_ratio']:.3f}",
                                    'Asset Diversity': f"{metrics['avg_asset_diversity']:.1f}",
                                    'Night Activity': f"{metrics['night_activity_ratio']:.3f}"
                                })
                        
                        behavioral_df = pd.DataFrame(behavioral_data)
                        
                        if len(behavioral_df) > 0:
                            st.dataframe(behavioral_df)
                            
                            # Create comparison charts
                            st.subheader("Behavioral Patterns Comparison")
                            
                            # Liquidation rate comparison
                            fig_liq = px.bar(
                                behavioral_df,
                                x='Range',
                                y=[float(x) for x in behavioral_df['Liquidation Rate']],
                                title='Liquidation Rate by Score Range'
                            )
                            fig_liq.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_liq, use_container_width=True)
                            
                            # Asset diversity comparison
                            fig_div = px.bar(
                                behavioral_df,
                                x='Range',
                                y=[float(x) for x in behavioral_df['Asset Diversity']],
                                title='Average Asset Diversity by Score Range'
                            )
                            fig_div.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_div, use_container_width=True)
                        
                        # Generate comprehensive analysis report
                        st.header("📋 Comprehensive Analysis Report")
                        
                        # Generate detailed analysis
                        def generate_analysis_report(scores_df, features_df, filtered_df, behavioral_df, range_df):
                            analysis_text = f"""
# DeFi Credit Scoring Analysis Report
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary
This analysis examines **{len(scores_df)}** wallets based on their Aave V2 transaction patterns from **{len(filtered_df)}** transactions. The scoring system evaluates wallet behavior across multiple dimensions including transaction volume, consistency, risk indicators, and diversification patterns.

## Dataset Overview
- **Total Wallets Analyzed**: {len(scores_df):,}
- **Total Transactions**: {len(filtered_df):,}
- **Average Score**: {scores_df['credit_score'].mean():.1f}
- **Median Score**: {scores_df['credit_score'].median():.1f}
- **Score Standard Deviation**: {scores_df['credit_score'].std():.1f}

## Score Distribution
{range_df.to_string(index=False)}

## Key Findings

### High-Risk Wallets (0-400)
**Population**: {len(scores_df[scores_df['credit_score'] < 400])} wallets ({(len(scores_df[scores_df['credit_score'] < 400])/len(scores_df)*100):.1f}% of total)

**Characteristics**:
- Higher liquidation rates indicating poor risk management
- Lower asset diversification suggesting concentration risk
- More frequent night-time activity potentially indicating automated behavior
- Irregular transaction patterns

### Medium-Risk Wallets (400-600)
**Population**: {len(scores_df[(scores_df['credit_score'] >= 400) & (scores_df['credit_score'] < 600)])} wallets ({(len(scores_df[(scores_df['credit_score'] >= 400) & (scores_df['credit_score'] < 600)])/len(scores_df)*100):.1f}% of total)

**Characteristics**:
- Moderate transaction consistency
- Balanced deposit/withdrawal ratios
- Average asset diversification
- Occasional liquidation events

### Low-Risk Wallets (600-1000)
**Population**: {len(scores_df[scores_df['credit_score'] >= 600])} wallets ({(len(scores_df[scores_df['credit_score'] >= 600])/len(scores_df)*100):.1f}% of total)

**Characteristics**:
- Excellent repayment history
- High asset diversification
- Consistent transaction patterns
- Minimal liquidation events

## Behavioral Analysis
{behavioral_df.to_string(index=False) if len(behavioral_df) > 0 else "No behavioral data available"}

## Risk Indicators
- **Liquidation Events**: {(filtered_df['action'] == 'liquidationcall').sum()} total liquidations
- **Night Activity**: {len(filtered_df[pd.to_datetime(filtered_df['timestamp'], unit='s').dt.hour.isin([0,1,2,3,4,5])])} transactions during night hours
- **Asset Concentration**: Average {filtered_df.groupby('userWallet')['assetSymbol'].nunique().mean():.1f} assets per wallet

## Recommendations
1. **Portfolio Managers**: Focus on wallets with scores above 600 for lending opportunities
2. **Risk Teams**: Monitor wallets below 400 for potential liquidation risks
3. **Product Teams**: Develop features to help medium-risk wallets improve their scores
4. **Compliance Teams**: Investigate wallets with suspicious night activity patterns

## Methodology Notes
- Scoring uses Random Forest with 50+ engineered features
- Business rules apply penalties for liquidations and suspicious activity
- Scores are scaled to 0-1000 range for interpretability
- Analysis is based on historical transaction patterns only
"""
                            return analysis_text
                        
                        # Generate and display analysis
                        analysis_report = generate_analysis_report(scores_df, features_df, filtered_df, behavioral_df, range_df)
                        
                        with st.expander("📊 Full Analysis Report", expanded=False):
                            st.markdown(analysis_report)
                        
                        # Save analysis to file
                        st.subheader("💾 Export Analysis")
                        
                        # Create downloadable analysis report
                        analysis_filename = f"wallet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        
                        st.download_button(
                            label="Download Full Analysis Report",
                            data=analysis_report,
                            file_name=analysis_filename,
                            mime="text/markdown",
                            help="Download the complete analysis report in Markdown format"
                        )
                        
                        # Score percentiles
                        st.subheader("Score Percentiles")
                        percentiles = [10, 25, 50, 75, 90, 95, 99]
                        perc_values = [scores_df['credit_score'].quantile(p/100) for p in percentiles]
                        
                        perc_df = pd.DataFrame({
                            'Percentile': [f"{p}th" for p in percentiles],
                            'Score': perc_values
                        })
                        
                        st.dataframe(perc_df)
                        
                        # Top and bottom wallets
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🏆 Top Scoring Wallets")
                            top_wallets = scores_df.nlargest(10, 'credit_score')[['userWallet', 'credit_score']]
                            st.dataframe(top_wallets)
                        
                        with col2:
                            st.subheader("⚠️ Lowest Scoring Wallets")
                            bottom_wallets = scores_df.nsmallest(10, 'credit_score')[['userWallet', 'credit_score']]
                            st.dataframe(bottom_wallets)
                        
                        # Wallet lookup
                        st.header("🔍 Wallet Analysis")
                        
                        selected_wallet = st.selectbox(
                            "Select a wallet to analyze:",
                            options=scores_df['userWallet'].tolist(),
                            help="Choose a wallet to see detailed analysis"
                        )
                        
                        if selected_wallet:
                            wallet_data = scores_df[scores_df['userWallet'] == selected_wallet].iloc[0]
                            wallet_transactions = filtered_df[filtered_df['userWallet'] == selected_wallet]
                            
                            # Wallet score and details
                            st.subheader(f"Wallet: {selected_wallet}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Credit Score", f"{wallet_data['credit_score']:.0f}")
                            with col2:
                                percentile = (scores_df['credit_score'] < wallet_data['credit_score']).mean() * 100
                                st.metric("Percentile", f"{percentile:.1f}%")
                            with col3:
                                st.metric("Total Transactions", len(wallet_transactions))
                            
                            # Transaction timeline
                            st.subheader("Transaction Timeline")
                            wallet_transactions_copy = wallet_transactions.copy()
                            wallet_transactions_copy['date'] = pd.to_datetime(wallet_transactions_copy['timestamp'], unit='s')
                            timeline_data = wallet_transactions_copy.groupby(['date', 'action']).size().reset_index(name='count')
                            
                            fig_timeline = px.line(
                                timeline_data,
                                x='date',
                                y='count',
                                color='action',
                                title='Transaction Activity Over Time'
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                            
                            # Action distribution
                            st.subheader("Action Distribution")
                            action_counts = wallet_transactions['action'].value_counts()
                            fig_actions = px.pie(
                                values=action_counts.values,
                                names=action_counts.index,
                                title='Transaction Types'
                            )
                            st.plotly_chart(fig_actions, use_container_width=True)
                        
                        # Download results
                        st.header("💾 Download Results")
                        
                        # Prepare download data
                        download_data = scores_df.copy()
                        csv_buffer = io.StringIO()
                        download_data.to_csv(csv_buffer, index=False)
                        csv_string = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="Download Credit Scores (CSV)",
                            data=csv_string,
                            file_name=f"credit_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Model explanation
                        st.header("📚 Model Explanation")
                        
                        with st.expander("Scoring Methodology"):
                            st.markdown("""
                            **Credit Score Calculation:**
                            
                            The credit score (0-1000) is calculated using a Random Forest ensemble model that analyzes:
                            
                            1. **Transaction Patterns**: Frequency, volume, and consistency of transactions
                            2. **Risk Behavior**: Liquidation events, borrowing patterns, and repayment history
                            3. **Portfolio Management**: Asset diversification and balance stability
                            4. **Time-based Analysis**: Activity consistency and seasonal patterns
                            5. **Protocol Interaction**: Depth of engagement with DeFi protocols
                            
                            **Score Interpretation:**
                            - **800-1000**: Excellent credit profile, consistent and responsible usage
                            - **600-799**: Good credit profile, reliable with minor risk factors
                            - **400-599**: Fair credit profile, moderate risk indicators
                            - **200-399**: Poor credit profile, significant risk factors
                            - **0-199**: Very poor credit profile, high-risk or bot-like behavior
                            """)
                        
                        with st.expander("Feature Engineering Details"):
                            st.markdown("""
                            **Key Features:**
                            
                            - **Volume Metrics**: Total transaction volume, average amounts, volume stability
                            - **Behavioral Patterns**: Deposit/withdrawal ratios, transaction timing, frequency
                            - **Risk Indicators**: Liquidation events, borrowing behavior, repayment patterns
                            - **Diversification**: Asset variety, protocol usage, network activity
                            - **Consistency**: Time-based activity patterns, transaction regularity
                            - **Efficiency**: Gas usage patterns, transaction success rates
                            """)
                    else:
                        st.error("Failed to train model and generate scores")
        
else:
    st.info("Please upload a JSON file containing Aave V2 transaction data or enable sample data to begin analysis.")
    
    # Show sample data format
    st.subheader("Expected Data Format")
    st.markdown("""
    The JSON file should contain an array of transaction objects with the following structure:
    
    ```json
    [
        {
            "userWallet": "0x...",
            "network": "polygon",
            "protocol": "aave_v2",
            "txHash": "0x...",
            "timestamp": 1629178166,
            "action": "deposit",
            "actionData": {
                "amount": "2000000000",
                "assetSymbol": "USDC",
                "assetPriceUSD": "0.99",
                "poolId": "0x..."
            }
        }
    ]
    ```
    """)
