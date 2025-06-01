import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up page config
st.set_page_config(
    page_title="Causal Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Import causal analysis modules
try:
    # Causal modeling
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    st.warning("pgmpy not available. Some causal graph features will be limited.")

# Time series and ML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr

class SalesDataGenerator:
    """Generate synthetic sales data with causal relationships"""

    def __init__(self, n_weeks=104, seed=42):
        self.n_weeks = n_weeks
        np.random.seed(seed)

    def generate_data(self):
        """Generate synthetic sales data with realistic causal structure"""
        dates = pd.date_range(start='2022-01-01', periods=self.n_weeks, freq='W')

        data = []
        base_price = 25.0
        base_sales = 100

        for i, date in enumerate(dates):
            week = i + 1

            # Seasonal patterns
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * week / 52)
            trend = 1 + 0.02 * week / 52

            # External factors
            competitor_price = base_price * (0.9 + 0.2 * np.random.random())
            market_conditions = np.random.normal(1, 0.1)

            # Marketing campaign (random events)
            has_campaign = np.random.random() < 0.2
            campaign_spend = np.random.exponential(800) if has_campaign else 0
            campaign_effect = 1 + (campaign_spend / 1000) * 0.1

            # Discount decisions (influenced by competitor price and seasonality)
            discount_probability = 0.1 + 0.2 * (competitor_price < base_price) + 0.1 * (seasonality < 1)
            has_discount = np.random.random() < discount_probability
            discount_percent = np.random.uniform(5, 30) if has_discount else 0

            actual_price = base_price * (1 - discount_percent / 100)

            # Price elasticity effect
            price_elasticity = -1.5
            price_effect = (actual_price / base_price) ** price_elasticity

            # Sales calculation with causal structure
            expected_sales = (base_sales *
                            seasonality *
                            trend *
                            price_effect *
                            campaign_effect *
                            market_conditions)

            # Add noise
            actual_sales = max(0, np.random.normal(expected_sales, expected_sales * 0.1))

            # Revenue and profit
            revenue = actual_sales * actual_price
            cost_per_unit = 15
            profit = actual_sales * (actual_price - cost_per_unit) - campaign_spend

            data.append({
                'date': date,
                'week': week,
                'sales': actual_sales,
                'price': actual_price,
                'base_price': base_price,
                'discount_percent': discount_percent,
                'has_discount': has_discount,
                'campaign_spend': campaign_spend,
                'has_campaign': has_campaign,
                'competitor_price': competitor_price,
                'seasonality': seasonality,
                'trend': trend,
                'revenue': revenue,
                'profit': profit,
                'market_conditions': market_conditions
            })

        return pd.DataFrame(data)

class CausalGraphBuilder:
    """Build and analyze causal graphs for sales data"""

    def __init__(self):
        self.model = None
        self.inference = None

    def build_sales_causal_graph(self, data):
        """Build causal graph representing sales relationships"""

        if not PGMPY_AVAILABLE:
            return self._build_simple_causal_model(data)

        # Create discretized versions for Bayesian Network
        df_discrete = self._discretize_variables(data)

        try:
            # Define the causal graph structure
            model = BayesianNetwork([
                ('competitor_price_cat', 'discount_cat'),
                ('seasonality_cat', 'discount_cat'),
                ('discount_cat', 'price_cat'),
                ('price_cat', 'sales_cat'),
                ('campaign_cat', 'sales_cat'),
                ('market_cat', 'sales_cat'),
                ('past_sales_cat', 'sales_cat')
            ])

            # Fit the model
            model.fit(df_discrete, estimator=MaximumLikelihoodEstimator)

            self.model = model
            self.inference = VariableElimination(model)

            return model
        except Exception as e:
            st.warning(f"Error building Bayesian Network: {e}")
            return self._build_simple_causal_model(data)

    def _discretize_variables(self, data):
        """Convert continuous variables to categorical for Bayesian Network"""
        df = data.copy()

        # Add lagged sales
        df['past_sales'] = df['sales'].shift(1).fillna(df['sales'].mean())

        # Discretize variables
        df['competitor_price_cat'] = pd.cut(df['competitor_price'],
                                          bins=3, labels=['low', 'medium', 'high'])
        df['seasonality_cat'] = pd.cut(df['seasonality'],
                                     bins=3, labels=['low', 'medium', 'high'])
        df['discount_cat'] = pd.cut(df['discount_percent'],
                                  bins=[-0.1, 0, 15, 100], labels=['none', 'low', 'high'])
        df['price_cat'] = pd.cut(df['price'],
                               bins=3, labels=['low', 'medium', 'high'])
        df['sales_cat'] = pd.cut(df['sales'],
                               bins=3, labels=['low', 'medium', 'high'])
        df['campaign_cat'] = pd.cut(df['campaign_spend'],
                                  bins=[-0.1, 0, 500, np.inf], labels=['none', 'low', 'high'])
        df['market_cat'] = pd.cut(df['market_conditions'],
                                bins=3, labels=['poor', 'average', 'good'])
        df['past_sales_cat'] = pd.cut(df['past_sales'],
                                    bins=3, labels=['low', 'medium', 'high'])

        # Convert to strings (required by pgmpy)
        categorical_cols = [col for col in df.columns if col.endswith('_cat')]
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        return df[categorical_cols]

    def _build_simple_causal_model(self, data):
        """Build simplified causal model using correlation analysis"""
        
        # Calculate key relationships
        causal_relationships = {}

        # Discount -> Sales relationship
        discount_sales_corr = data['discount_percent'].corr(data['sales'])
        causal_relationships['discount_to_sales'] = discount_sales_corr

        # Campaign -> Sales relationship
        campaign_sales_corr = data['campaign_spend'].corr(data['sales'])
        causal_relationships['campaign_to_sales'] = campaign_sales_corr

        # Price -> Sales relationship
        price_sales_corr = data['price'].corr(data['sales'])
        causal_relationships['price_to_sales'] = price_sales_corr

        # Competitor price -> Discount decision
        comp_discount_corr = data['competitor_price'].corr(data['discount_percent'])
        causal_relationships['competitor_to_discount'] = comp_discount_corr

        self.simple_model = causal_relationships
        return causal_relationships

    def analyze_causal_effects(self, data):
        """Analyze causal effects using the built graph"""
        if not PGMPY_AVAILABLE or self.model is None:
            return self._analyze_simple_causal_effects(data)

        results = {}

        try:
            # Query: Effect of discount on sales
            prob_high_sales_with_discount = self.inference.query(
                variables=['sales_cat'],
                evidence={'discount_cat': 'high'}
            )

            prob_high_sales_no_discount = self.inference.query(
                variables=['sales_cat'],
                evidence={'discount_cat': 'none'}
            )

            results['discount_effect'] = {
                'with_discount': prob_high_sales_with_discount,
                'without_discount': prob_high_sales_no_discount
            }

        except Exception as e:
            st.warning(f"Error in causal analysis: {e}")
            return self._analyze_simple_causal_effects(data)

        return results

    def _analyze_simple_causal_effects(self, data):
        """Simplified causal effect analysis using statistical methods"""
        results = {}

        # Compare sales with and without discounts
        discount_data = data[data['has_discount'] == True]
        no_discount_data = data[data['has_discount'] == False]

        if len(discount_data) > 0 and len(no_discount_data) > 0:
            # T-test for difference in means
            t_stat, p_value = stats.ttest_ind(discount_data['sales'], no_discount_data['sales'])

            results['discount_effect'] = {
                'with_discount_mean': discount_data['sales'].mean(),
                'without_discount_mean': no_discount_data['sales'].mean(),
                'difference': discount_data['sales'].mean() - no_discount_data['sales'].mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        # Campaign effect analysis
        campaign_data = data[data['has_campaign'] == True]
        no_campaign_data = data[data['has_campaign'] == False]

        if len(campaign_data) > 0 and len(no_campaign_data) > 0:
            t_stat_camp, p_value_camp = stats.ttest_ind(campaign_data['sales'], no_campaign_data['sales'])

            results['campaign_effect'] = {
                'with_campaign_mean': campaign_data['sales'].mean(),
                'without_campaign_mean': no_campaign_data['sales'].mean(),
                'difference': campaign_data['sales'].mean() - no_campaign_data['sales'].mean(),
                't_statistic': t_stat_camp,
                'p_value': p_value_camp,
                'significant': p_value_camp < 0.05
            }

        return results

class SalesForecastingModel:
    """Advanced forecasting model with causal features"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}

    def prepare_features(self, data):
        """Engineer features for forecasting"""
        df = data.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Lag features
        for lag in [1, 2, 4, 8]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            df[f'discount_lag_{lag}'] = df['discount_percent'].shift(lag)

        # Rolling statistics
        for window in [4, 8, 12]:
            df[f'sales_ma_{window}'] = df['sales'].rolling(window).mean()
            df[f'sales_std_{window}'] = df['sales'].rolling(window).std()
            df[f'discount_ma_{window}'] = df['discount_percent'].rolling(window).mean()

        # Time features
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Interaction features
        df['price_discount_interaction'] = df['price'] * df['discount_percent']
        df['campaign_discount_interaction'] = df['campaign_spend'] * df['discount_percent']

        # Price elasticity proxy
        df['price_change'] = df['price'].pct_change()
        df['sales_change'] = df['sales'].pct_change()


        return df

    def train_ensemble_model(self, data, target_col='sales', test_size=0.2):
        df = self.prepare_features(data)
        df = df.dropna() 

        feature_cols = [col for col in df.columns if col not in ['date', 'sales', 'revenue', 'profit'] ]
        
        X = df[feature_cols]
        y = df[target_col]

        # Train-test split (temporal)
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

            # Save feature names to use during inference
            # This is important and you already have it.
            # No need to assign feature_names_in_ if it's already an attribute of the fitted model.
            # If a model doesn't have it (like LinearRegression), you might need to handle it.
            if not hasattr(model, 'feature_names_in_'):
                model.feature_names_in_ = X_train.columns.to_numpy() # Convert to numpy array for compatibility
            
            y_pred = model.predict(X_test)

            self.models[name] = model
            self.metrics[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }

            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df


        ensemble_pred = np.mean([model.predict(X_test) for model in models.values()], axis=0)
        self.metrics['ensemble'] = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'mape': np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        }

        return X_train, X_test, y_train, y_test


    def forecast_with_intervention(self, data, intervention_params, n_periods=4):
        # Start with the full prepared historical data
        prepared_historical_df = self.prepare_features(data) #

        forecasts = {}

        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                continue

            predictions = []
            
            # Get the exact feature names the model was trained on
            # This is crucial for matching during prediction
            model_features = getattr(model, 'feature_names_in_', []) #

            # Start prediction from the last known historical data point
            # Ensure this row has all the necessary lagged features from history
            current_forecast_df = prepared_historical_df.iloc[-1:].copy() #

            for period in range(n_periods):
                # 1. Update exogenous variables based on intervention params
                current_forecast_df['discount_percent'] = intervention_params.get('discount_percent', 0) #
                current_forecast_df['campaign_spend'] = intervention_params.get('campaign_spend', 0) #
                
                # 2. Update time features for the next period
                last_date = current_forecast_df['date'].iloc[0] #
                next_date = last_date + timedelta(weeks=1) #
                current_forecast_df['date'] = next_date #
                current_forecast_df['month'] = next_date.month #
                current_forecast_df['quarter'] = next_date.quarter #
                current_forecast_df['week_of_year'] = next_date.isocalendar().week 
                if period == 0:
                    # For the very first forecast period, the lags are from the *actual* historical data.
                    # current_forecast_df already represents the last row of `prepared_historical_df`.
                    pass # `current_forecast_df` is already good
                else:
                
                    temp_df_for_lags = prepared_historical_df.copy()

            

                if period == 0:
                    # For the first forecast step, the input features are simply the last row
                    # of the prepared historical data.
                    X_pred_row = prepared_historical_df.iloc[-1:].copy()
                    
                    # Apply intervention params to this row
                    X_pred_row['discount_percent'] = intervention_params.get('discount_percent', X_pred_row['discount_percent'].iloc[0])
                    X_pred_row['campaign_spend'] = intervention_params.get('campaign_spend', X_pred_row['campaign_spend'].iloc[0])

                    # Update time features
                    last_hist_date = prepared_historical_df['date'].iloc[-1]
                    next_date = last_hist_date + timedelta(weeks=1)
                    X_pred_row['date'] = next_date
                    X_pred_row['month'] = next_date.month
                    X_pred_row['quarter'] = next_date.quarter
                    X_pred_row['week_of_year'] = next_date.isocalendar().week

                    # For the first step, sales_lag_1 (and others) are already correctly set from `prepared_historical_df`
                    # `X_pred_row` now contains the features for the first forecast period.
                    
                else:
                    # For subsequent steps (period > 0), we use the *last predicted sales*
                    # to generate the new lagged features.
                    # We need enough historical context + previous predictions to compute lags.
                    
                    # Create a temporary DataFrame with enough past context to compute lags
                    # Combine original data tail (e.g., last 15-20 rows) + already made predictions
                    # The max lag is 8, max rolling window is 12. Let's take last 20 rows of original data.
                    
                    # Create a temporary DataFrame for this step
                    temp_df_for_lags_and_rolling = pd.DataFrame()
                    if period == 1: # For the second forecast, we need the first forecast's output
                        # Combine actual historical data (last 20 rows) with the first forecast row
                        temp_df_for_lags_and_rolling = pd.concat([data.tail(20), pd.DataFrame([forecast_rows[-1]])], ignore_index=True)
                    else: # For subsequent forecasts, combine historical data with all previous forecast rows
                        temp_df_for_lags_and_rolling = pd.concat([data.tail(20), pd.DataFrame(forecast_rows)], ignore_index=True)

                    # Now, run `prepare_features` on this combined temporary DataFrame
                    temp_prepared = self.prepare_features(temp_df_for_lags_and_rolling)

                    # Extract the last row, which contains the features for the *current* forecast period
                    X_pred_row = temp_prepared.iloc[-1:].copy()
                    
                    # Apply intervention parameters for the current forecast period
                    X_pred_row['discount_percent'] = intervention_params.get('discount_percent', 0)
                    X_pred_row['campaign_spend'] = intervention_params.get('campaign_spend', 0)

                    # Update time features for the current forecast period
                    last_known_date = X_pred_row['date'].iloc[0] # This would be previous forecast's date
                    next_date = last_known_date + timedelta(weeks=1)
                    X_pred_row['date'] = next_date
                    X_pred_row['month'] = next_date.month
                    X_pred_row['quarter'] = next_date.quarter
                    X_pred_row['week_of_year'] = next_date.isocalendar().week
                
                # Ensure only the necessary features are passed and in the correct order
                X_pred_final = X_pred_row.reindex(columns=model_features) #
                
                # Fill any remaining NaNs in the prediction row.
                # If your `prepare_features` consistently handles NaNs by dropping (in training),
                # you must ensure the prediction input doesn't have them in critical features.
                # The previous `fillna(0)` could cause issues if the model learned from non-zero values.
                # If `dropna()` was used in training, then make sure these prediction rows don't have NaNs.
                # Given how lags are generated, the first few forecast periods might have NaNs if history is short.
                # Best practice: use 0 or mean/median of training data for NaNs introduced at the beginning of prediction.
                X_pred_final = X_pred_final.fillna(0) # or replace with mean/median from X_train

                # Make the prediction
                pred = model.predict(X_pred_final)[0] #
                predictions.append(pred) #

                # Update the `current_forecast_df` (or `forecast_rows` for next iteration)
                # with the predicted sales for the *next* iteration's lag features.
                # Create a record of this predicted period for use in next iteration's lags
                predicted_row_dict = X_pred_row.iloc[0].to_dict()
                predicted_row_dict['sales'] = pred # Update sales with the prediction
                predicted_row_dict['revenue'] = pred * predicted_row_dict['price'] # Update revenue based on predicted sales
                predicted_row_dict['profit'] = pred * (predicted_row_dict['price'] - 15) - predicted_row_dict['campaign_spend'] # Update profit

                # Initialize `forecast_rows` list if it's the first prediction
                if period == 0:
                    forecast_rows = [predicted_row_dict]
                else:
                    forecast_rows.append(predicted_row_dict)

            forecasts[model_name] = predictions

        # Ensemble forecast
        ensemble_forecast = np.mean(list(forecasts.values()), axis=0) #
        forecasts['ensemble'] = ensemble_forecast #

        return forecasts #

class CounterfactualAnalyzer:
    """Perform counterfactual analysis for causal inference"""

    def __init__(self, forecasting_model):
        self.forecasting_model = forecasting_model

    def estimate_counterfactual_sales(self, data, actual_intervention, counterfactual_intervention, n_periods=4):
        """Estimate what would have happened under different intervention"""

        # Forecast with actual intervention
        actual_forecast = self.forecasting_model.forecast_with_intervention(
            data, actual_intervention, n_periods
        )

        # Forecast with counterfactual intervention
        counterfactual_forecast = self.forecasting_model.forecast_with_intervention(
            data, counterfactual_intervention, n_periods
        )

        # Calculate causal impact
        causal_impact = {}
        for model_name in actual_forecast.keys():
            actual_sales = np.array(actual_forecast[model_name])
            counterfactual_sales = np.array(counterfactual_forecast[model_name])

            impact = actual_sales - counterfactual_sales
            relative_impact = impact / counterfactual_sales * 100

            causal_impact[model_name] = {
                'absolute_impact': impact,
                'relative_impact': relative_impact,
                'total_impact': np.sum(impact),
                'average_impact': np.mean(relative_impact)
            }

        return {
            'actual_forecast': actual_forecast,
            'counterfactual_forecast': counterfactual_forecast,
            'causal_impact': causal_impact
        }

    def sensitivity_analysis(self, data, base_intervention, parameter_ranges, n_periods=4):
        """Perform sensitivity analysis on intervention parameters"""

        results = {}

        for param_name, param_range in parameter_ranges.items():
            param_results = []

            for param_value in param_range:
                intervention = base_intervention.copy()
                intervention[param_name] = param_value

                forecast = self.forecasting_model.forecast_with_intervention(
                    data, intervention, n_periods
                )

                total_sales = np.sum(forecast['ensemble'])
                param_results.append({
                    'parameter_value': param_value,
                    'total_forecasted_sales': total_sales,
                    'forecast': forecast['ensemble']
                })

            results[param_name] = param_results

        return results
    
    
def main():
    st.markdown('<h1 class="main-header">📊 Causal Sales Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.title("🔧 Controls")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_weeks = st.sidebar.slider("Number of weeks", 52, 208, 104)
    seed = st.sidebar.number_input("Random seed", value=42, min_value=1)
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    
    # Generate data button
    if st.sidebar.button("🔄 Generate New Data") or not st.session_state.data_generated:
        with st.spinner("Generating synthetic sales data..."):
            data_generator = SalesDataGenerator(n_weeks=n_weeks, seed=seed)
            st.session_state.data = data_generator.generate_data()
            st.session_state.data_generated = True
        st.success("Data generated successfully!")
    
    if st.session_state.data_generated:
        data = st.session_state.data
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Data Overview", 
            "🔗 Causal Analysis", 
            "🔮 Forecasting", 
            "🎲 Counterfactual Analysis",
            "📊 Insights & Recommendations"
        ])
        
        with tab1:
            st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Weeks", len(data))
                st.metric("Avg Weekly Sales", f"{data['sales'].mean():.1f}")
            
            with col2:
                st.metric("Avg Discount Rate", f"{data['discount_percent'].mean():.1f}%")
                st.metric("Weeks with Discounts", f"{data['has_discount'].sum()}")
            
            with col3:
                st.metric("Total Campaign Spend", f"${data['campaign_spend'].sum():.0f}")
                st.metric("Avg Weekly Revenue", f"${data['revenue'].mean():.0f}")
            
            with col4:
                st.metric("Avg Weekly Profit", f"${data['profit'].mean():.0f}")
                st.metric("Max Weekly Sales", f"{data['sales'].max():.1f}")
            
            # Data visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sales over time
                fig = px.line(data, x='date', y='sales', title='📈 Sales Over Time')
                fig.update_traces(line=dict(width=3))
                fig.update_layout(title_font=dict(size=18), xaxis_title='Date', yaxis_title='Sales')
                st.plotly_chart(fig, use_container_width=True)
                
                # Price vs Sales scatter
                fig = px.scatter(data, x='price', y='sales', color='discount_percent',
                 title='🎯 Price vs Sales (Colored by Discount %)', color_continuous_scale='Viridis')
                fig.update_layout(title_font=dict(size=18), xaxis_title='Price', yaxis_title='Sales')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Discount impact comparison
                hist_data = [
                go.Histogram(x=data[data['has_discount']]['sales'], name='With Discount', opacity=0.75),
                go.Histogram(x=data[~data['has_discount']]['sales'], name='No Discount', opacity=0.75)
                ]
                fig = go.Figure(data=hist_data)
                fig.update_layout(barmode='overlay', title='🏷️ Sales Distribution by Discount Status',
                  xaxis_title='Sales', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
                
                # Campaign effect
                fig = px.box(data, x='has_campaign', y='sales',
             title='📣 Sales by Campaign Status', labels={'has_campaign': 'Campaign Active'})
                fig.update_layout(title_font=dict(size=18), yaxis_title='Sales')
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            corr = data[['sales', 'discount_percent', 'campaign_spend', 'price', 'profit']].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu',
                title='📊 Correlation Heatmap', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data preview
            st.subheader("Data Sample")
            st.dataframe(data.head(10))
        
        with tab2:
            st.markdown('<div class="section-header">Causal Analysis</div>', unsafe_allow_html=True)
            
            with st.spinner("Building causal graph and analyzing effects..."):
                # Build causal graph
                causal_builder = CausalGraphBuilder()
                causal_model = causal_builder.build_sales_causal_graph(data)
                causal_effects = causal_builder.analyze_causal_effects(data)
            
            st.success("Causal analysis completed!")
            
            # Display causal relationships
            if hasattr(causal_builder, 'simple_model'):
                st.subheader("Causal Relationships (Correlations)")
                
                relationships_df = pd.DataFrame(list(causal_builder.simple_model.items()), 
                                              columns=['Relationship', 'Correlation'])
                st.dataframe(relationships_df)
                
                # Visualize causal relationships
                fig, ax = plt.subplots(figsize=(10, 6))
                relationships_df.set_index('Relationship')['Correlation'].plot(kind='bar', ax=ax)
                ax.set_title('Causal Relationships Strength', fontsize=14, fontweight='bold')
                ax.set_ylabel('Correlation Coefficient')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            # Causal effects analysis
            if causal_effects:
                st.subheader("Causal Effects Analysis")
    
                if 'discount_effect' in causal_effects:
                    discount_effect = causal_effects['discount_effect']
        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sales with Discount", f"{discount_effect.get('with_discount_mean', 0):.1f}")
                        st.metric("Sales without Discount", f"{discount_effect.get('without_discount_mean', 0):.1f}")
        
                    with col2:
                        st.metric("Difference", f"{discount_effect.get('difference', 0):.1f}")
                        significance = (
                             "✅ Significant"
                                if discount_effect.get('significant', False)
                                else "❌ Not Significant"
                        )
                        st.metric("Statistical Significance", significance)

                if 'campaign_effect' in causal_effects:
                    campaign_effect = causal_effects['campaign_effect']
        
                    st.subheader("Campaign Effect")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sales with Campaign", f"{campaign_effect.get('with_campaign_mean', 0):.1f}")
                        st.metric("Sales without Campaign", f"{campaign_effect.get('without_campaign_mean', 0):.1f}")
        
                    with col2:
                        st.metric("Difference", f"{campaign_effect.get('difference', 0):.1f}")
                        significance = (
                            "✅ Significant"
                            if campaign_effect.get('significant', False)
                            else "❌ Not Significant"
                        )
                st.metric("Statistical Significance", significance)

        
        with tab3:
            st.markdown('<div class="section-header">Sales Forecasting</div>', unsafe_allow_html=True)
            
            # Train forecasting models
            if st.button("🚀 Train Forecasting Models"):
                with st.spinner("Training ensemble forecasting models..."):
                    forecasting_model = SalesForecastingModel()
                    X_train, X_test, y_train, y_test = forecasting_model.train_ensemble_model(data)
                    st.session_state.forecasting_model = forecasting_model
                st.success("Models trained successfully!")
            
            if 'forecasting_model' in st.session_state:
                forecasting_model = st.session_state.forecasting_model
                
                # Model performance metrics
                st.subheader("Model Performance")
                metrics_df = pd.DataFrame(forecasting_model.metrics).T
                st.dataframe(metrics_df.round(2))
                
                # Feature importance
                if forecasting_model.feature_importance:
                    st.subheader("Feature Importance (Random Forest)")
                    importance_df = forecasting_model.feature_importance['random_forest'].head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['feature'], importance_df['importance'])
                    ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Importance')
                    st.pyplot(fig)
                
                # Forecasting interface
                st.subheader("Generate Forecast")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    discount_percent = st.slider("Discount %", 0, 30, 15)
                with col2:
                    campaign_spend = st.slider("Campaign Spend", 0, 2000, 500)
                with col3:
                    n_periods = st.slider("Forecast Periods", 1, 12, 4)
                
                if st.button("📈 Generate Forecast"):
                    intervention_params = {
                        'discount_percent': discount_percent,
                        'campaign_spend': campaign_spend
                    }
                    
                    with st.spinner("Generating forecast..."):
                        forecasts = forecasting_model.forecast_with_intervention(
                            data, intervention_params, n_periods
                        )
                    
                    st.success("Forecast generated!")
                    
                    # Display forecast results
                    forecast_df = pd.DataFrame(forecasts)
                    forecast_df.index = [f"Week {i+1}" for i in range(n_periods)]
                    st.dataframe(forecast_df.round(1))
                    
                    # Visualize forecast
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data (last 20 weeks)
                    recent_data = data.tail(20)
                    ax.plot(range(-len(recent_data), 0), recent_data['sales'], 
                           'o-', label='Historical Sales', linewidth=2, markersize=4)
                    
                    # Plot forecasts
                    forecast_range = range(1, n_periods + 1)
                    for model_name, preds in forecasts.items():
                        if model_name == 'ensemble':
                            ax.plot(forecast_range, preds, 's-', label=f'{model_name.title()} Forecast', 
                                   linewidth=3, markersize=6)
                        else:
                            ax.plot(forecast_range, preds, '--', alpha=0.7, 
                                   label=f'{model_name.replace("_", " ").title()}')
                    
                    ax.axvline(x=0, color='red', linestyle=':', alpha=0.7, label='Forecast Start')
                    ax.set_title(f'Sales Forecast (Discount: {discount_percent}%, Campaign: ${campaign_spend})', 
                                fontsize=14, fontweight='bold')
                    ax.set_xlabel('Weeks (relative to present)')
                    ax.set_ylabel('Sales')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        with tab4:
            st.markdown('<div class="section-header">Counterfactual Analysis</div>', unsafe_allow_html=True)
            
            if 'forecasting_model' not in st.session_state:
                st.warning("Please train forecasting models in the Forecasting tab first.")
            else:
                forecasting_model = st.session_state.forecasting_model
                
                st.subheader("What-If Analysis")
                st.write("Compare different intervention strategies to understand causal impact.")
                
                # Counterfactual analysis interface
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Scenario A (Actual)")
                    discount_a = st.slider("Discount % (A)", 0, 30, 20, key="discount_a")
                    campaign_a = st.slider("Campaign Spend (A)", 0, 2000, 1000, key="campaign_a")
                
                with col2:
                    st.subheader("Scenario B (Counterfactual)")
                    discount_b = st.slider("Discount % (B)", 0, 30, 5, key="discount_b")
                    campaign_b = st.slider("Campaign Spend (B)", 0, 2000, 200, key="campaign_b")
                
                n_periods_cf = st.slider("Analysis Periods", 1, 8, 4, key="cf_periods")
                
                if st.button("🔍 Run Counterfactual Analysis"):
                    actual_intervention = {
                        'discount_percent': discount_a,
                        'campaign_spend': campaign_a
                    }
                    
                    counterfactual_intervention = {
                        'discount_percent': discount_b,
                        'campaign_spend': campaign_b
                    }
                    
                    with st.spinner("Running counterfactual analysis..."):
                        counterfactual_analyzer = CounterfactualAnalyzer(forecasting_model)
                        cf_results = counterfactual_analyzer.estimate_counterfactual_sales(
                            data, actual_intervention, counterfactual_intervention, n_periods_cf
                        )
                    
                    st.success("Counterfactual analysis completed!")
                    
                    # Display results
                    st.subheader("Causal Impact Results")
                    
                    # Summary metrics
                    ensemble_impact = cf_results['causal_impact']['ensemble']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Impact", f"{ensemble_impact['total_impact']:.1f} units")
                    with col2:
                        st.metric("Average Impact", f"{ensemble_impact['average_impact']:.1f}%")
                    with col3:
                        impact_direction = "📈 Positive" if ensemble_impact['total_impact'] > 0 else "📉 Negative"
                        st.metric("Impact Direction", impact_direction)
                    
                    # Detailed comparison table
                    comparison_data = []
                    for period in range(n_periods_cf):
                        comparison_data.append({
                            'Period': f'Week {period + 1}',
                            'Scenario A': cf_results['actual_forecast']['ensemble'][period],
                            'Scenario B': cf_results['counterfactual_forecast']['ensemble'][period],
                            'Difference': ensemble_impact['absolute_impact'][period],
                            'Relative Impact (%)': ensemble_impact['relative_impact'][period]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.round(2))
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Forecast comparison
                    periods = range(1, n_periods_cf + 1)
                    ax1.plot(periods, cf_results['actual_forecast']['ensemble'], 
                            'o-', label='Scenario A', linewidth=2, markersize=6)
                    ax1.plot(periods, cf_results['counterfactual_forecast']['ensemble'], 
                            's-', label='Scenario B', linewidth=2, markersize=6)
                    ax1.set_title('Sales Forecast Comparison', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Week')
                    ax1.set_ylabel('Sales')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Impact visualization
                    colors = ['green' if x > 0 else 'red' for x in ensemble_impact['absolute_impact']]
                    ax2.bar(periods, ensemble_impact['absolute_impact'], color=colors, alpha=0.7)
                    ax2.set_title('Causal Impact by Period', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Week')
                    ax2.set_ylabel('Sales Impact')
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Model-wise impact comparison
                    st.subheader("Impact by Model")
                    model_impacts = []
                    for model_name, impact_data in cf_results['causal_impact'].items():
                        model_impacts.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Total Impact': impact_data['total_impact'],
                            'Average Impact (%)': impact_data['average_impact']
                        })
                    
                    model_impact_df = pd.DataFrame(model_impacts)
                    st.dataframe(model_impact_df.round(2))
        
        with tab5:
            st.markdown('<div class="section-header">Insights & Recommendations</div>', unsafe_allow_html=True)
            
            # Key insights from the analysis
            st.subheader("📋 Key Insights")
            
            # Price elasticity insight
            price_sales_corr = data['price'].corr(data['sales'])
            st.write(f"**Price Elasticity**: Sales and price correlation is {price_sales_corr:.3f}")
            
            if price_sales_corr < -0.3:
                st.success("✅ Strong negative correlation suggests customers are price-sensitive. Discounting strategies can be effective.")
            elif price_sales_corr > -0.1:
                st.warning("⚠️ Weak price sensitivity detected. Focus on value proposition rather than price competition.")
            
            # Discount effectiveness
            if 'forecasting_model' in st.session_state:
                discount_data = data[data['has_discount'] == True]['sales']
                no_discount_data = data[data['has_discount'] == False]['sales']
                
                if len(discount_data) > 0 and len(no_discount_data) > 0:
                    discount_lift = (discount_data.mean() - no_discount_data.mean()) / no_discount_data.mean() * 100
                    st.write(f"**Discount Effectiveness**: Average sales lift from discounting is {discount_lift:.1f}%")
                    
                    if discount_lift > 10:
                        st.success("✅ Discounting is highly effective for driving sales.")
                    elif discount_lift > 0:
                        st.info("ℹ️ Discounting provides moderate sales lift.")
                    else:
                        st.error("❌ Discounting may be hurting sales. Review discount strategy.")
            
            # Campaign ROI analysis
            campaign_data = data[data['has_campaign'] == True]
            if len(campaign_data) > 0:
                avg_campaign_spend = campaign_data['campaign_spend'].mean()
                avg_campaign_sales = campaign_data['sales'].mean()
                no_campaign_sales = data[data['has_campaign'] == False]['sales'].mean()
                
                sales_lift = avg_campaign_sales - no_campaign_sales
                # Assuming $20 profit per unit
                campaign_roi = (sales_lift * 20 - avg_campaign_spend) / avg_campaign_spend * 100
                
                st.write(f"**Campaign ROI**: Estimated ROI is {campaign_roi:.1f}%")
                
                if campaign_roi > 50:
                    st.success("✅ Campaigns are highly profitable. Consider increasing campaign frequency.")
                elif campaign_roi > 0:
                    st.info("ℹ️ Campaigns are profitable but could be optimized.")
                else:
                    st.error("❌ Campaigns may not be cost-effective. Review targeting and creative.")
            
            # Seasonality insights
            data['month'] = data['date'].dt.month
            monthly_sales = data.groupby('month')['sales'].mean()
            peak_month = monthly_sales.idxmax()
            low_month = monthly_sales.idxmin()
            
            st.write(f"**Seasonality**: Peak sales in month {peak_month}, lowest in month {low_month}")
            
            # Recommendations
            st.subheader("🎯 Strategic Recommendations")
            
            recommendations = []
            
            # Price-based recommendations
            if price_sales_corr < -0.4:
                recommendations.append("Consider dynamic pricing strategies to maximize revenue during low-demand periods.")
            
            # Discount recommendations
            discount_freq = data['has_discount'].mean()
            if discount_freq > 0.3:
                recommendations.append("High discount frequency detected. Consider reducing frequency to avoid training customers to wait for sales.")
            elif discount_freq < 0.1:
                recommendations.append("Low discount frequency. Consider strategic discounting to drive volume during slow periods.")
            
            # Campaign recommendations
            if 'campaign_roi' in locals() and campaign_roi > 50:
                recommendations.append("High campaign ROI suggests opportunity to increase marketing investment.")
            elif 'campaign_roi' in locals() and campaign_roi < 20:
                recommendations.append("Low campaign ROI indicates need for campaign optimization or budget reallocation.")
            
            # Seasonal recommendations
            seasonality_variance = data.groupby(data['date'].dt.month)['sales'].mean().std()
            if seasonality_variance > data['sales'].std() * 0.3:
                recommendations.append("Strong seasonal patterns detected. Plan inventory and campaigns around peak months.")
            
            # General recommendations
            recommendations.extend([
                "Use counterfactual analysis regularly to test intervention strategies before implementation.",
                "Monitor competitor pricing closely as it influences discount decisions.",
                "Consider A/B testing different campaign spend levels to optimize ROI."
            ])
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Future analysis suggestions
            st.subheader("🔮 Future Analysis Opportunities")
            
            future_analyses = [
                "Customer segmentation analysis to understand different price sensitivities",
                "Multi-touch attribution modeling to better measure campaign effectiveness",
                "Competitor response modeling to predict competitive reactions",
                "Inventory optimization using demand forecasts",
                "Cross-product analysis to understand portfolio effects"
            ]
            
            for i, analysis in enumerate(future_analyses, 1):
                st.write(f"{i}. {analysis}")
            
            # Export functionality
            st.subheader("📁 Export Results")
            
            if st.button("📊 Generate Summary Report"):
                # Create summary statistics
                summary_stats = {
                    'Metric': [
                        'Average Weekly Sales',
                        'Average Price',
                        'Average Discount %',
                        'Campaign Weeks',
                        'Total Revenue',
                        'Average Weekly Profit'
                    ],
                    'Value': [
                        f"{data['sales'].mean():.1f}",
                        f"${data['price'].mean():.2f}",
                        f"{data['discount_percent'].mean():.1f}%",
                        f"{data['has_campaign'].sum()}",
                        f"${data['revenue'].sum():.0f}",
                        f"${data['profit'].mean():.0f}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_stats)
                
                # Convert to CSV
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary CSV",
                    data=csv,
                    file_name=f"sales_analysis_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Full data export
                csv_full = data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset CSV",
                    data=csv_full,
                    file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()