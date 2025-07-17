# DeFi Credit Scoring System

## Overview

This is a machine learning-powered credit scoring system that analyzes Aave V2 transaction patterns to assign wallet creditworthiness scores from 0-1000. The system processes DeFi transaction data to evaluate wallet creditworthiness using advanced feature engineering and machine learning techniques, with higher scores indicating reliable and responsible usage.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit-based Web Interface**: Interactive dashboard for data analysis and visualization
- **Real-time Processing**: One-step script for immediate scoring from JSON data
- **Visualization Components**: Plotly charts for interactive data exploration
- **File Upload Interface**: Direct JSON file upload capability

### Backend Architecture
- **Modular Python Architecture**: Three main processing components
  - `DataProcessor`: Handles raw transaction data cleaning and standardization
  - `FeatureEngineer`: Extracts 50+ meaningful features from transaction patterns
  - `CreditScoringModel`: Implements Random Forest ensemble learning for credit assessment
- **Unsupervised Learning Approach**: Creates target variables from transaction patterns without labeled data
- **Real-time Scoring**: Processes uploaded data immediately without persistence

### Data Storage
- **File-based Input**: JSON transaction data upload
- **In-memory Processing**: No persistent database, processes data in memory
- **Sample Data**: Includes embedded sample dataset for testing

## Key Components

### 1. Data Processing Pipeline (`data_processor.py`)
- **Purpose**: Cleans and standardizes raw Aave V2 transaction data
- **Functions**: Data validation, action extraction, derived field creation
- **Input**: Raw JSON transaction data from Aave V2 protocol
- **Output**: Structured DataFrame with cleaned transaction records

### 2. Feature Engineering (`feature_engineering.py`)
- **Purpose**: Extracts meaningful features from transaction patterns
- **Capabilities**: 
  - Wallet-level behavioral analysis
  - Interaction feature creation
  - Feature scaling and normalization
  - Missing value handling
- **Output**: Feature matrix with wallet addresses as index

### 3. Credit Scoring Model (`credit_scoring_model.py`)
- **Algorithm**: Random Forest Regressor
- **Approach**: Unsupervised scoring with synthetic target variable creation
- **Features**: Cross-validation, feature importance analysis, model metrics
- **Output**: Credit scores from 0-1000 with feature importance rankings

### 4. Streamlit Application (`app.py`)
- **Interface**: Web-based dashboard with file upload and visualization
- **Controls**: Configurable model parameters (estimators, test size, minimum transactions)
- **Visualization**: Interactive charts and model performance metrics
- **Port**: Configured to run on port 5000

## Data Flow

1. **Data Input**: JSON files containing Aave V2 transaction data uploaded via Streamlit interface
2. **Data Processing**: Raw transactions cleaned and standardized by DataProcessor
3. **Feature Engineering**: Transaction patterns converted to numerical features by FeatureEngineer
4. **Model Training**: Random Forest model trained on engineered features using unsupervised approach
5. **Score Generation**: Credit scores (0-1000) generated for each wallet
6. **Visualization**: Results displayed through interactive Streamlit dashboard

## External Dependencies

### Core Libraries
- **Streamlit**: Web interface framework
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Plotly**: Interactive visualization and charting
- **Matplotlib**: Additional plotting capabilities

### Data Requirements
- **Transaction Data**: Aave V2 protocol transactions in JSON format
- **Required Fields**: userWallet, network, protocol, txHash, timestamp, action, actionData
- **Supported Networks**: Polygon (primary), extensible to other networks
- **Transaction Types**: Deposits, withdrawals, borrowing, repayments, liquidations

## Deployment Strategy

### Local Development
- **Setup**: Python 3.8+ environment with automatic dependency installation
- **Execution**: Single command `streamlit run app.py --server.port 5000`
- **Access**: Web interface at `http://localhost:5000`

### Production Considerations
- **Scalability**: Currently designed for single-user analysis, would need session management for multi-user
- **Performance**: In-memory processing suitable for moderate dataset sizes
- **Security**: No authentication layer, file uploads processed directly
- **Monitoring**: Basic logging implemented in each component

### Key Architectural Decisions

1. **Unsupervised Learning**: Chosen due to lack of labeled credit score data in DeFi space
2. **Random Forest**: Selected for robustness and interpretability in financial scoring
3. **Feature Engineering Focus**: Emphasized behavioral pattern extraction over raw transaction volume
4. **Streamlit Interface**: Chosen for rapid prototyping and easy deployment
5. **Modular Design**: Separated concerns for maintainability and testing
6. **In-memory Processing**: Simplified architecture for immediate results without persistence overhead

The system is designed as a proof-of-concept for DeFi credit scoring with emphasis on interpretability and ease of use rather than production-scale deployment.