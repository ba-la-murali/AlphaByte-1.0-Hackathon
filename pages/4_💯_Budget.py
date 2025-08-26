
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Technical Analysis
import ta
from huggingface_hub import InferenceClient

class HuggingFaceLlamaLLM:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=1500, temperature=0.1):
        self.client = InferenceClient(
            provider="fireworks-ai",
            api_key=api_key,
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def __call__(self, prompt):
        return self.generate(prompt)
    
    def generate(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling Hugging Face API: {str(e)}")
            return "Sorry, I encountered an error processing your request."

class AdvancedStockAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def calculate_advanced_features(self, stock_data):
        """Calculate comprehensive technical indicators and features"""
        df = stock_data.copy()
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove ticker level
        
        # Ensure column names are strings and handle common variations
        df.columns = [str(col).strip() for col in df.columns]
        
        # Handle different column name variations
        price_col = None
        volume_col = None
        
        # Find price column
        for col in df.columns:
            if 'adj close' in col.lower() or col.lower() == 'adj close':
                price_col = col
                break
            elif 'close' in col.lower() and price_col is None:
                price_col = col
        
        # Find volume column
        for col in df.columns:
            if 'volume' in col.lower():
                volume_col = col
                break
        
        if price_col is None:
            raise ValueError("No suitable price column found")
        
        # Standardize column names
        df['Adj Close'] = df[price_col].squeeze()  # squeeze() ensures single column
        df['High'] = df[[col for col in df.columns if 'high' in col.lower()][0]].squeeze()
        df['Low'] = df[[col for col in df.columns if 'low' in col.lower()][0]].squeeze()
        df['Open'] = df[[col for col in df.columns if 'open' in col.lower()][0]].squeeze()
        
        if volume_col:
            df['Volume'] = df[volume_col].squeeze()
        else:
            # Create dummy volume if not available
            df['Volume'] = pd.Series(1000000, index=df.index)
            st.warning("Volume data not available, using dummy values")
        
        # Ensure all price columns are numeric
        for col in ['Adj Close', 'High', 'Low', 'Open']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any remaining NaN values in price data
        df = df.dropna(subset=['Adj Close', 'High', 'Low', 'Open'])
        
        if len(df) < 50:
            raise ValueError("Insufficient data after cleaning")
        
        try:
            # Price-based features with error handling
            df['MA_5'] = df['Adj Close'].rolling(5, min_periods=5).mean()
            df['MA_10'] = df['Adj Close'].rolling(10, min_periods=10).mean()
            df['MA_20'] = df['Adj Close'].rolling(20, min_periods=20).mean()
            df['MA_50'] = df['Adj Close'].rolling(50, min_periods=50).mean()
            df['MA_200'] = df['Adj Close'].rolling(200, min_periods=100).mean()
            
            # Moving Average Ratios with safe division
            df['MA5_MA20_Ratio'] = np.where(
                df['MA_20'] != 0, 
                df['MA_5'] / df['MA_20'], 
                1.0
            )
            df['MA20_MA50_Ratio'] = np.where(
                df['MA_50'] != 0, 
                df['MA_20'] / df['MA_50'], 
                1.0
            )
            df['Price_MA20_Ratio'] = np.where(
                df['MA_20'] != 0, 
                df['Adj Close'] / df['MA_20'], 
                1.0
            )
            
            # Technical Indicators with try-catch for each
            try:
                df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close'], window=14).rsi()
            except:
                df['RSI'] = 50.0  # Neutral RSI
                
            try:
                df['RSI_30'] = ta.momentum.RSIIndicator(df['Adj Close'], window=30).rsi()
            except:
                df['RSI_30'] = 50.0
            
            # MACD with error handling
            try:
                macd = ta.trend.MACD(df['Adj Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()
            except:
                df['MACD'] = 0.0
                df['MACD_Signal'] = 0.0
                df['MACD_Histogram'] = 0.0
            
            # Bollinger Bands with error handling
            try:
                bb = ta.volatility.BollingerBands(df['Adj Close'], window=20)
                df['BB_Upper'] = bb.bollinger_hband()
                df['BB_Lower'] = bb.bollinger_lband()
                df['BB_Width'] = np.where(
                    df['Adj Close'] != 0,
                    (df['BB_Upper'] - df['BB_Lower']) / df['Adj Close'],
                    0.0
                )
                df['BB_Position'] = np.where(
                    (df['BB_Upper'] - df['BB_Lower']) != 0,
                    (df['Adj Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']),
                    0.5
                )
            except:
                df['BB_Upper'] = df['Adj Close'] * 1.02
                df['BB_Lower'] = df['Adj Close'] * 0.98
                df['BB_Width'] = 0.04
                df['BB_Position'] = 0.5
            
            # Stochastic Oscillator with error handling
            try:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Adj Close'])
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
            except:
                df['Stoch_K'] = 50.0
                df['Stoch_D'] = 50.0
            
            # Williams %R with error handling
            try:
                df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Adj Close']).williams_r()
            except:
                df['Williams_R'] = -50.0
            
            # Average Directional Index (ADX) with error handling
            try:
                df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Adj Close']).adx()
            except:
                df['ADX'] = 25.0
            
            # Commodity Channel Index (CCI) with error handling
            try:
                df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Adj Close']).cci()
            except:
                df['CCI'] = 0.0
            
            # Price momentum features
            df['Price_Change_1d'] = df['Adj Close'].pct_change(1)
            df['Price_Change_5d'] = df['Adj Close'].pct_change(5)
            df['Price_Change_10d'] = df['Adj Close'].pct_change(10)
            df['Price_Change_20d'] = df['Adj Close'].pct_change(20)
            
            # Volatility features
            df['Volatility_5d'] = df['Price_Change_1d'].rolling(5, min_periods=5).std()
            df['Volatility_20d'] = df['Price_Change_1d'].rolling(20, min_periods=20).std()
            df['Volatility_50d'] = df['Price_Change_1d'].rolling(50, min_periods=50).std()
            
            # Volume features with safe operations
            if 'Volume' in df.columns:
                df['Volume_MA_5'] = df['Volume'].rolling(5, min_periods=5).mean()
                df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=20).mean()
                df['Volume_Ratio_5'] = np.where(
                    df['Volume_MA_5'] != 0,
                    df['Volume'] / df['Volume_MA_5'],
                    1.0
                )
                df['Volume_Ratio_20'] = np.where(
                    df['Volume_MA_20'] != 0,
                    df['Volume'] / df['Volume_MA_20'],
                    1.0
                )
            else:
                df['Volume_Ratio_5'] = 1.0
                df['Volume_Ratio_20'] = 1.0
            
            # Price-Volume features
            df['PV_Trend'] = df['Price_Change_1d'] * df['Volume_Ratio_5']
            
            # High-Low features with safe division
            df['High_Low_Ratio'] = np.where(
                df['Low'] != 0,
                df['High'] / df['Low'],
                1.0
            )
            df['Close_High_Ratio'] = np.where(
                df['High'] != 0,
                df['Adj Close'] / df['High'],
                1.0
            )
            df['Close_Low_Ratio'] = np.where(
                df['Low'] != 0,
                df['Adj Close'] / df['Low'],
                1.0
            )
            
            # Gap features
            df['Gap'] = np.where(
                df['Adj Close'].shift(1) != 0,
                (df['Open'] - df['Adj Close'].shift(1)) / df['Adj Close'].shift(1),
                0.0
            )
            
            # Trend strength
            df['Trend_20d'] = np.where(
                df['Adj Close'] > df['Adj Close'].shift(20), 1, 
                np.where(df['Adj Close'] < df['Adj Close'].shift(20), -1, 0)
            )
            
            # Fill any remaining NaN values with appropriate defaults
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col.endswith('_Ratio'):
                    df[col] = df[col].fillna(1.0)
                elif 'RSI' in col or 'Stoch' in col:
                    df[col] = df[col].fillna(50.0)
                elif 'MACD' in col or 'CCI' in col or 'Williams' in col:
                    df[col] = df[col].fillna(0.0)
                elif 'Volatility' in col:
                    df[col] = df[col].fillna(0.02)
                else:
                    df[col] = df[col].fillna(df[col].median())
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error calculating technical features: {str(e)}")

    
    def create_target_variable(self, df, prediction_days=5, return_threshold=0.02):
        """Create target variable based on future returns"""
        # Calculate future returns
        df['Future_Return'] = df['Close'].shift(-prediction_days) / df['Close'] - 1
        
        # Create multi-class target: 0=Hold, 1=Buy, 2=Strong Buy
        conditions = [
            df['Future_Return'] < -return_threshold,  # Strong Sell/Hold
            (df['Future_Return'] >= -return_threshold) & (df['Future_Return'] < return_threshold),  # Hold
            (df['Future_Return'] >= return_threshold) & (df['Future_Return'] < return_threshold*2),  # Buy
            df['Future_Return'] >= return_threshold*2  # Strong Buy
        ]
        choices = [0, 0, 1, 2]  # 0=Hold, 1=Buy, 2=Strong Buy
        
        df['Target'] = np.select(conditions, choices, default=0)
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix for ML models"""
        feature_columns = [
            'MA5_MA20_Ratio', 'MA20_MA50_Ratio', 'Price_MA20_Ratio',
            'RSI', 'RSI_30', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'BB_Position', 'Stoch_K', 'Stoch_D', 'Williams_R',
            'ADX', 'CCI', 'Price_Change_1d', 'Price_Change_5d', 'Price_Change_10d', 'Price_Change_20d',
            'Volatility_5d', 'Volatility_20d', 'Volatility_50d',
            'Volume_Ratio_5', 'Volume_Ratio_20', 'PV_Trend',
            'High_Low_Ratio', 'Close_High_Ratio', 'Close_Low_Ratio', 'Gap', 'Trend_20d'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        return df[available_features], df['Target'] if 'Target' in df.columns else None
    
    def get_ml_models(self):
        """Define available ML models"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'Ensemble Voting': None  # Will be created dynamically
        }
        return models
    
    def train_model(self, stocks, model_name, risk_factor, progress_bar=None):
        """Train selected ML model"""
        all_features = []
        all_targets = []
        successful_stocks = []
        
        for i, stock_symbol in enumerate(stocks):
            if progress_bar:
                progress_bar.progress((i + 1) / len(stocks))
            
            try:
                # Download stock data
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=1095)  # 3 years
                
                stock_data = self.download_stock_data_safely(stock_symbol, start_date, end_date)
                
                # Validate data quality
                if len(stock_data) < 100:
                    st.warning(f"Insufficient data for {stock_symbol} ({len(stock_data)} days)")
                    continue
                    
                # Check for required columns
                if stock_data.empty or len(stock_data.columns) < 4:
                    st.warning(f"Invalid data structure for {stock_symbol}")
                    continue
                
                # Calculate features with error handling
                df = self.calculate_advanced_features(stock_data)
                df = self.create_target_variable(df, prediction_days=5, 
                                            return_threshold=0.015 if risk_factor == 'Low' 
                                            else 0.025 if risk_factor == 'Medium' else 0.035)
                
                # Prepare features
                features, targets = self.prepare_features(df)
                
                # Remove NaN values
                valid_mask = ~(features.isna().any(axis=1) | targets.isna())
                features_clean = features[valid_mask]
                targets_clean = targets[valid_mask]
                
                if len(features_clean) > 50:
                    all_features.append(features_clean)
                    all_targets.append(targets_clean)
                    successful_stocks.append(stock_symbol)
                    st.success(f"✅ Successfully processed {stock_symbol}")
                else:
                    st.warning(f"Insufficient clean data for {stock_symbol}")
                    
            except Exception as e:
                st.warning(f"❌ Could not process {stock_symbol}: {str(e)}")
                continue
        
        # Check if we have enough data to train
        if len(all_features) == 0:
            st.error("❌ No stocks could be processed successfully")
            return None, None, None
        elif len(all_features) < 2:
            st.warning(f"⚠️ Only {len(all_features)} stock(s) processed. Results may be unreliable.")
        
        st.info(f"✅ Successfully processed {len(successful_stocks)} out of {len(stocks)} stocks: {', '.join(successful_stocks)}")
            
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Handle class imbalance
        from collections import Counter
        class_counts = Counter(y)
        st.info(f"Class distribution: Hold={class_counts[0]}, Buy={class_counts[1]}, Strong Buy={class_counts.get(2, 0)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # Get and train model
        models = self.get_ml_models()
        
        if model_name == 'Ensemble Voting':
            # Create ensemble of top 3 models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            model = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xgb_model), ('gb', gb)],
                voting='soft'
            )
        else:
            model = models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        evaluation_metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cross_val_score(model, X_scaled, y, cv=5).mean() if len(np.unique(y)) > 1 else 0
        }
        
        return model, scaler, evaluation_metrics
    
    def get_prediction(self, stock_symbol, model, scaler, risk_factor):
        """Get prediction for a single stock"""
        try:
            # Download recent data
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=365)
            
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
            
            if len(stock_data) < 50:
                return self._default_prediction()
            
            # Calculate features
            df = self.calculate_advanced_features(stock_data)
            features, _ = self.prepare_features(df)
            
            # Get latest features
            latest_features = features.iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return self._default_prediction()
            
            # Scale and predict
            latest_features_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_features_scaled)[0]
            probabilities = model.predict_proba(latest_features_scaled)[0]
            
            # Risk adjustment
            volatility = df['Volatility_20d'].iloc[-1] if 'Volatility_20d' in df.columns else 0.02
            risk_adjustment = self._apply_risk_filter(prediction, probabilities, volatility, risk_factor)
            
            recommendation_map = {0: 'Hold', 1: 'Buy', 2: 'Strong Buy'}
            
            return {
                'prediction': prediction,
                'recommendation': recommendation_map.get(risk_adjustment, 'Hold'),
                'confidence': max(probabilities),
                'probabilities': {
                    'Hold': probabilities[0],
                    'Buy': probabilities[1] if len(probabilities) > 1 else 0,
                    'Strong Buy': probabilities[2] if len(probabilities) > 2 else 0
                },
                'volatility': volatility,
                'risk_score': self._calculate_risk_score(volatility, risk_factor)
            }
            
        except Exception as e:
            st.warning(f"Error predicting {stock_symbol}: {str(e)}")
            return self._default_prediction()
    
    def _default_prediction(self):
        """Return default prediction when analysis fails"""
        return {
            'prediction': 0,
            'recommendation': 'Hold',
            'confidence': 0.5,
            'probabilities': {'Hold': 0.6, 'Buy': 0.3, 'Strong Buy': 0.1},
            'volatility': 0.02,
            'risk_score': 'Unknown'
        }
    def download_stock_data_safely(self, symbol, start_date, end_date):
        """Safely download stock data with multiple attempts"""
        attempts = [
            lambda: yf.download(symbol, start=start_date, end=end_date, progress=False),
            lambda: yf.Ticker(symbol).history(start=start_date, end=end_date),
            lambda: yf.download(symbol, start=start_date, end=end_date, progress=False, threads=False)
        ]
        
        for i, download_func in enumerate(attempts):
            try:
                data = download_func()
                
                if data.empty:
                    continue
                    
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    # Flatten MultiIndex - keep only the first level (remove ticker)
                    data.columns = data.columns.droplevel(1)
                
                # Ensure we have minimum required data
                if len(data) >= 50:
                    return data
                    
            except Exception as e:
                if i == len(attempts) - 1:  # Last attempt
                    raise Exception(f"All download attempts failed for {symbol}: {str(e)}")
                continue
        
        raise Exception(f"Could not download sufficient data for {symbol}")

    def _apply_risk_filter(self, prediction, probabilities, volatility, risk_factor):
        """Apply risk-based filtering to predictions"""
        risk_thresholds = {
            'Low': {'vol_threshold': 0.025, 'conf_threshold': 0.8},
            'Medium': {'vol_threshold': 0.04, 'conf_threshold': 0.7},
            'High': {'vol_threshold': 0.1, 'conf_threshold': 0.6}
        }
        
        threshold = risk_thresholds[risk_factor]
        
        # Conservative approach for low risk
        if risk_factor == 'Low':
            if volatility > threshold['vol_threshold']:
                return 0  # Force Hold for high volatility
            elif prediction > 0 and max(probabilities) < threshold['conf_threshold']:
                return 0  # Require high confidence
        
        # Moderate filtering for medium risk
        elif risk_factor == 'Medium':
            if volatility > threshold['vol_threshold'] and prediction == 2:
                return 1  # Downgrade Strong Buy to Buy for high volatility
        
        # Minimal filtering for high risk (allow original prediction)
        
        return prediction
    
    def _calculate_risk_score(self, volatility, risk_preference):
        """Calculate risk compatibility score"""
        if volatility < 0.02:
            risk_level = "Low Risk"
        elif volatility < 0.035:
            risk_level = "Medium Risk"
        elif volatility < 0.05:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        compatibility = {
            ('Low', 'Low Risk'): '✅ Perfect Match',
            ('Low', 'Medium Risk'): '⚠️ Slightly Higher Risk',
            ('Low', 'High Risk'): '❌ Too Risky',
            ('Low', 'Very High Risk'): '❌ Much Too Risky',
            ('Medium', 'Low Risk'): '✅ Conservative Choice',
            ('Medium', 'Medium Risk'): '✅ Good Match',
            ('Medium', 'High Risk'): '⚠️ Higher Risk',
            ('Medium', 'Very High Risk'): '❌ Too Risky',
            ('High', 'Low Risk'): '⚠️ Conservative',
            ('High', 'Medium Risk'): '✅ Reasonable',
            ('High', 'High Risk'): '✅ Good Match',
            ('High', 'Very High Risk'): '⚠️ Very Aggressive'
        }
        
        return compatibility.get((risk_preference, risk_level), risk_level)

def create_advanced_visualizations(stocks_data, recommendations):
    """Create comprehensive visualizations"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Stock Prices Comparison', 'Recommendation Distribution', 
                       'Risk vs Confidence Analysis', 'Volatility Analysis'),
        specs=[[{"secondary_y": True}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Stock prices comparison
    for stock, data in stocks_data.items():
        if 'price_history' in data:
            fig.add_trace(
                go.Scatter(
                    x=data['price_history'].index,
                    y=data['price_history']['Close'],
                    name=stock,
                    mode='lines'
                ),
                row=1, col=1
            )
    
    # Recommendation distribution
    rec_counts = pd.Series([rec['recommendation'] for rec in recommendations.values()]).value_counts()
    fig.add_trace(
        go.Pie(
            labels=rec_counts.index,
            values=rec_counts.values,
            name="Recommendations"
        ),
        row=1, col=2
    )
    
    # Risk vs Confidence scatter
    for stock, rec in recommendations.items():
        fig.add_trace(
            go.Scatter(
                x=[rec['volatility']],
                y=[rec['confidence']],
                mode='markers+text',
                text=[stock],
                textposition="top center",
                name=stock,
                marker=dict(
                    size=15,
                    color='green' if rec['recommendation'] == 'Buy' 
                    else 'blue' if rec['recommendation'] == 'Strong Buy'
                    else 'orange'
                )
            ),
            row=2, col=1
        )
    
    # Volatility analysis
    volatilities = [rec['volatility'] for rec in recommendations.values()]
    stock_names = list(recommendations.keys())
    
    fig.add_trace(
        go.Bar(
            x=stock_names,
            y=volatilities,
            name="Volatility",
            marker_color=['red' if v > 0.04 else 'yellow' if v > 0.025 else 'green' for v in volatilities]
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="📊 Advanced Stock Analysis Dashboard"
    )
    
    return fig

def generate_advanced_prompt(recommendations, risk_factor, investment_amount, model_metrics, selected_model):
    """Generate comprehensive analysis prompt"""
    
    risk_strategies = {
        "Low": {
            "strategy": "Capital preservation with steady income generation",
            "allocation": "60% large-cap stable stocks, 30% dividend-paying stocks, 10% bonds/cash",
            "stop_loss": "5-7%",
            "position_size": "Equal weight distribution",
            "rebalancing": "Quarterly review"
        },
        "Medium": {
            "strategy": "Balanced growth with moderate risk management",
            "allocation": "70% growth stocks, 20% value stocks, 10% speculative",
            "stop_loss": "8-12%",
            "position_size": "Weight by conviction and volatility",
            "rebalancing": "Monthly review"
        },
        "High": {
            "strategy": "Aggressive growth seeking maximum returns",
            "allocation": "80% growth/momentum stocks, 15% small-cap, 5% cash",
            "stop_loss": "15-20%",
            "position_size": "Concentrated positions in high conviction picks",
            "rebalancing": "Weekly review"
        }
    }
    
    strategy = risk_strategies[risk_factor]
    
    # Calculate portfolio metrics
    total_stocks = len(recommendations)
    buy_stocks = len([r for r in recommendations.values() if r['recommendation'] in ['Buy', 'Strong Buy']])
    avg_confidence = np.mean([r['confidence'] for r in recommendations.values()])
    avg_volatility = np.mean([r['volatility'] for r in recommendations.values()])
    
    prompt = f"""
    As a senior quantitative analyst, provide a comprehensive investment analysis using advanced ML model insights.
    
    PORTFOLIO OVERVIEW:
    - Investment Amount: ${investment_amount:,.2f}
    - Risk Profile: {risk_factor}
    - ML Model Used: {selected_model}
    - Model Accuracy: {model_metrics.get('test_accuracy', 0):.3f}
    - Cross-validation Score: {model_metrics.get('cv_scores', 0):.3f}
    
    STOCK ANALYSIS RESULTS:
    {recommendations}
    
    PORTFOLIO METRICS:
    - Total Stocks Analyzed: {total_stocks}
    - Recommended for Purchase: {buy_stocks}
    - Average Model Confidence: {avg_confidence:.3f}
    - Portfolio Volatility: {avg_volatility:.4f}
    
    RISK MANAGEMENT STRATEGY:
    - Strategy: {strategy['strategy']}
    - Recommended Allocation: {strategy['allocation']}
    - Stop Loss Level: {strategy['stop_loss']}
    - Position Sizing: {strategy['position_size']}
    - Rebalancing: {strategy['rebalancing']}
    
    PROVIDE DETAILED ANALYSIS INCLUDING:
    1. Executive Summary: Key insights and overall portfolio assessment
    2. Individual Stock Analysis: Brief analysis of each stock's ML prediction and risk profile
    3. Portfolio Construction: Specific allocation percentages for ${investment_amount:,.2f}
    4. Risk Management: Entry points, stop-losses, and position sizing
    5. Market Timing: Current market conditions and optimal entry strategy
    6. Expected Returns: Realistic return expectations with timeline
    7. Alternative Recommendations: Suggest improvements or alternatives
    8. Model Confidence Assessment: Interpret ML model reliability
    
    Keep analysis professional, data-driven, and actionable. Limit to 12-15 comprehensive lines.
    """
    
    return prompt

# Streamlit App
def main():
    st.set_page_config(
        page_title="Advanced AI Stock Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 Advanced AI Stock Analyzer</h1>', unsafe_allow_html=True)
    # st.markdown("### *Powered by Machine Learning & Technical Analysis*")
    
    # Initialize components
    analyzer = AdvancedStockAnalyzer()
    HF_TOKEN = ""
    
    if not HF_TOKEN:
        st.error("⚠️ Please set HF_TOKEN environment variable")
        return
    
    llm = HuggingFaceLlamaLLM(api_key=HF_TOKEN)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("📊 Analysis Configuration")
        
        # Investment parameters
        investment_amount = st.number_input(
            "💰 Investment Amount ($)",
            min_value=100,
            value=10000,
            step=500,
            help="Enter your total investment amount"
        )
        
        # Stock symbols
        stock_symbols = st.text_input(
            "📈 Stock Symbols",
            value="AAPL,GOOGL,MSFT,TSLA",
            help="Enter comma-separated stock symbols (e.g., AAPL,GOOGL,MSFT)"
        )
        
        # Risk factor
        risk_factor = st.selectbox(
            "⚖️ Risk Tolerance",
            ["Low", "Medium", "High"],
            index=1,
            help="Select your risk tolerance level"
        )
        
        # ML Model selection
        ml_model = st.selectbox(
            "🤖 ML Model",
            ["Random Forest", "XGBoost", "Gradient Boosting", "SVM", 
             "Neural Network", "Logistic Regression", "Ensemble Voting"],
            index=0,
            help="Choose the machine learning model for predictions"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            prediction_horizon = st.selectbox(
                "Prediction Horizon",
                [3, 5, 7, 10],
                index=1,
                help="Days ahead to predict"
            )
            
            min_confidence = st.slider(
                "Minimum Confidence Threshold",
                0.5, 0.95, 0.7,
                help="Minimum model confidence for buy recommendations"
            )
            
            enable_visualizations = st.checkbox(
                "Enable Advanced Visualizations",
                value=True
            )
    
    # Main Analysis Section
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        if stock_symbols:
            stocks_list = [s.strip().upper() for s in stock_symbols.split(",")]
            st.metric("Stocks to Analyze", len(stocks_list))
            st.metric("Investment Amount", f"${investment_amount:,}")
            st.metric("Risk Level", risk_factor)
            st.metric("ML Model", ml_model)
    
    # Analysis Button
    if st.button("🚀 Run Advanced Analysis", type="primary"):
        if not stock_symbols.strip():
            st.error("❌ Please enter stock symbols")
            return
        
        stocks = [symbol.strip().upper() for symbol in stock_symbols.split(",")]
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            st.info("🔄 Starting advanced analysis...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Phase 1: Data Collection
            status_text.text("📊 Collecting market data...")
            stocks_data = {}
            
            for i, stock in enumerate(stocks):
                try:
                    # Get current price and basic info
                    ticker = yf.Ticker(stock)
                    info = ticker.info
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    price_history = ticker.history(period="6mo")
                    
                    stocks_data[stock] = {
                        'current_price': current_price,
                        'info': info,
                        'price_history': price_history
                    }
                except:
                    st.warning(f"⚠️ Could not fetch data for {stock}")
                
                progress_bar.progress((i + 1) / (len(stocks) * 3))
            
            # Phase 2: ML Model Training
            status_text.text(f"🤖 Training {ml_model} model...")
            model, scaler, metrics = analyzer.train_model(stocks, ml_model, risk_factor, progress_bar)
            
            if model is None:
                st.error("❌ Failed to train ML model")
                return
            
    # Phase 3: Generate Predictions
            status_text.text("🔮 Generating predictions...")
            recommendations = {}
            
            for i, stock in enumerate(stocks):
                prediction_result = analyzer.get_prediction(stock, model, scaler, risk_factor)
                
                recommendations[stock] = {
                    'recommendation': prediction_result['recommendation'],
                    'confidence': prediction_result['confidence'],
                    'probabilities': prediction_result['probabilities'],
                    'volatility': prediction_result['volatility'],
                    'risk_score': prediction_result['risk_score'],
                    'current_price': stocks_data.get(stock, {}).get('current_price', 0)
                }
                
                progress_bar.progress((len(stocks) * 2 + i + 1) / (len(stocks) * 3))
            
            # Clear progress indicators
            progress_container.empty()
            
            # Display Results
            st.markdown("## 📊 Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                buy_count = len([r for r in recommendations.values() if r['recommendation'] in ['Buy', 'Strong Buy']])
                st.metric("🟢 Buy Signals", buy_count, f"{buy_count/len(stocks)*100:.1f}%")
            
            with col2:
                avg_confidence = np.mean([r['confidence'] for r in recommendations.values()])
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}", f"{avg_confidence*100:.1f}%")
            
            with col3:
                model_accuracy = metrics.get('test_accuracy', 0)
                st.metric("🤖 Model Accuracy", f"{model_accuracy:.3f}", f"{model_accuracy*100:.1f}%")
            
            with col4:
                avg_volatility = np.mean([r['volatility'] for r in recommendations.values()])
                volatility_level = "Low" if avg_volatility < 0.025 else "Medium" if avg_volatility < 0.04 else "High"
                st.metric("📈 Portfolio Volatility", f"{avg_volatility:.4f}", volatility_level)
            
            # Detailed Recommendations Table
            st.markdown("### 🎯 ML-Powered Recommendations")
            
            # Create recommendations DataFrame
            rec_data = []
            for stock, rec in recommendations.items():
                rec_data.append({
                    'Stock': stock,
                    'Recommendation': rec['recommendation'],
                    'Confidence': f"{rec['confidence']:.3f}",
                    'Current Price': f"${rec['current_price']:.2f}",
                    'Volatility': f"{rec['volatility']:.4f}",
                    'Risk Assessment': rec['risk_score'],
                    'Hold Prob': f"{rec['probabilities']['Hold']:.2f}",
                    'Buy Prob': f"{rec['probabilities']['Buy']:.2f}",
                    'Strong Buy Prob': f"{rec['probabilities']['Strong Buy']:.2f}"
                })
            
            rec_df = pd.DataFrame(rec_data)
            
            # Color-code recommendations
            def color_recommendations(val):
                if val == 'Strong Buy':
                    return 'background-color: #90EE90'
                elif val == 'Buy':
                    return 'background-color: #98FB98'
                else:
                    return 'background-color: #FFE4B5'
            
            styled_df = rec_df.style.applymap(color_recommendations, subset=['Recommendation'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Model Performance Metrics
            with st.expander("🤖 Model Performance Details"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Accuracy", f"{metrics.get('train_accuracy', 0):.4f}")
                
                with col2:
                    st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
                
                with col3:
                    st.metric("Cross-Validation Score", f"{metrics.get('cv_scores', 0):.4f}")
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_') and len(analyzer.feature_names) > 0:
                    st.markdown("#### 📈 Feature Importance")
                    
                    feature_imp = pd.DataFrame({
                        'Feature': analyzer.feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        feature_imp, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            
            # Advanced Visualizations
            if enable_visualizations:
                st.markdown("### 📊 Advanced Visualizations")
                
                try:
                    viz_fig = create_advanced_visualizations(stocks_data, recommendations)
                    st.plotly_chart(viz_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create advanced visualizations: {str(e)}")
                
                # Individual stock charts
                st.markdown("#### 📈 Individual Stock Analysis")
                
                selected_stock = st.selectbox("Select stock for detailed analysis:", stocks)
                
                if selected_stock in stocks_data and 'price_history' in stocks_data[selected_stock]:
                    stock_data = stocks_data[selected_stock]['price_history']
                    rec = recommendations[selected_stock]
                    
                    # Create candlestick chart
                    fig_stock = go.Figure(data=go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name=selected_stock
                    ))
                    
                    # Add moving averages
                    fig_stock.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(20).mean(),
                        name='MA20',
                        line=dict(color='orange')
                    ))
                    
                    fig_stock.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'].rolling(50).mean(),
                        name='MA50',
                        line=dict(color='blue')
                    ))
                    
                    fig_stock.update_layout(
                        title=f"{selected_stock} - {rec['recommendation']} (Confidence: {rec['confidence']:.3f})",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=500
                    )
                    
                    st.plotly_chart(fig_stock, use_container_width=True)
                    
                    # Stock details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${rec['current_price']:.2f}")
                        st.metric("Volatility", f"{rec['volatility']:.4f}")
                    
                    with col2:
                        st.metric("ML Confidence", f"{rec['confidence']:.3f}")
                        st.metric("Risk Assessment", rec['risk_score'])
                    
                    with col3:
                        stock_info = stocks_data[selected_stock].get('info', {})
                        market_cap = stock_info.get('marketCap', 0)
                        if market_cap:
                            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
                        
                        pe_ratio = stock_info.get('forwardPE', 0)
                        if pe_ratio:
                            st.metric("Forward P/E", f"{pe_ratio:.1f}")
            
            # AI-Generated Analysis
            st.markdown("### 🧠 AI Investment Analysis")
            
            with st.spinner("🤖 Generating comprehensive analysis..."):
                enhanced_prompt = generate_advanced_prompt(
                    recommendations, risk_factor, investment_amount, metrics, ml_model
                )
                
                ai_analysis = llm(enhanced_prompt)
            
            st.markdown("#### 📝 Professional Investment Report")
            st.write(ai_analysis)
            
            # Portfolio Allocation Suggestion
            st.markdown("### 💼 Suggested Portfolio Allocation")
            
            # Calculate suggested allocation
            buy_stocks = {k: v for k, v in recommendations.items() if v['recommendation'] in ['Buy', 'Strong Buy']}
            
            if buy_stocks:
                # Weight by confidence and inverse volatility
                weights = {}
                total_weight = 0
                
                for stock, rec in buy_stocks.items():
                    # Higher weight for higher confidence and lower volatility
                    weight = rec['confidence'] * (1 / (rec['volatility'] + 0.01))
                    if rec['recommendation'] == 'Strong Buy':
                        weight *= 1.5  # Boost strong buy signals
                    
                    weights[stock] = weight
                    total_weight += weight
                
                # Normalize weights
                allocation_data = []
                remaining_amount = investment_amount
                
                for stock, weight in weights.items():
                    allocation_pct = (weight / total_weight) * 100
                    allocation_amount = (weight / total_weight) * investment_amount
                    
                    allocation_data.append({
                        'Stock': stock,
                        'Allocation %': f"{allocation_pct:.1f}%",
                        'Amount': f"${allocation_amount:,.2f}",
                        'Shares (approx)': int(allocation_amount / recommendations[stock]['current_price']),
                        'Recommendation': recommendations[stock]['recommendation']
                    })
                
                allocation_df = pd.DataFrame(allocation_data)
                st.dataframe(allocation_df, use_container_width=True)
                
                # Pie chart of allocation
                fig_pie = px.pie(
                    allocation_df, 
                    values=[float(x.strip('$').replace(',', '')) for x in allocation_df['Amount']],
                    names='Stock',
                    title="Suggested Portfolio Allocation"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("⚠️ No stocks recommended for purchase based on current analysis.")
                st.info("💡 Consider adjusting your risk tolerance or analyzing different stocks.")
            
            # Risk Analysis Summary
            st.markdown("### ⚖️ Risk Analysis Summary")
            
            risk_summary = []
            for stock, rec in recommendations.items():
                risk_level = "Low" if rec['volatility'] < 0.025 else "Medium" if rec['volatility'] < 0.04 else "High"
                risk_summary.append({
                    'Stock': stock,
                    'Volatility': f"{rec['volatility']:.4f}",
                    'Risk Level': risk_level,
                    'Risk Score': rec['risk_score'],
                    'Suitable for Profile': "✅" if any(x in rec['risk_score'] for x in ["Perfect", "Good", "Match"]) else "⚠️"
                })
            
            risk_df = pd.DataFrame(risk_summary)
            st.dataframe(risk_df, use_container_width=True)
            
            # Export Results
            st.markdown("### 📥 Export Analysis")
            
            # Prepare export data
            export_data = {
                'analysis_date': datetime.datetime.now().isoformat(),
                'investment_amount': investment_amount,
                'risk_factor': risk_factor,
                'ml_model': ml_model,
                'model_metrics': metrics,
                'recommendations': recommendations,
                'ai_analysis': ai_analysis
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download CSV Report"):
                    csv_data = rec_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"stock_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📄 Download JSON Report"):
                    json_data = pd.Series(export_data).to_json(indent=2)
                    st.download_button(
                        label="💾 Download JSON",
                        data=json_data,
                        file_name=f"stock_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.error("Please check your stock symbols and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This analysis is for educational purposes only. 
    Always consult with a financial advisor before making investment decisions.</p>

    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
   main()
