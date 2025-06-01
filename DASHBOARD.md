# Causal Sales Analysis Dashboard

## 🚀 Project Overview & Uniqueness

This project stands out as a complete **end-to-end causal inference and sales analytics platform** that goes far beyond typical dashboard applications. It uniquely combines:

* ✅ Synthetic Data Generation with Realistic Causal Structures
* 🔗 Bayesian Network-based Causal Modeling
* 📈 Advanced Time Series Forecasting with Causal Features
* 🎲 Counterfactual Analysis for Business Decision Making
* 📊 Interactive Visualization and Strategic Recommendations

---

## 🛠️ Detailed Technical Breakdown

### 1. 📦 Import Dependencies & Setup

#### **Core Libraries Used**

**Streamlit Components:**

* `streamlit`: Modern web app framework for data science
**Data Science Stack:**

* `pandas`, `numpy`: Data manipulation & numerical computing
* `matplotlib.pyplot`, `seaborn`: Static plotting
* `plotly.express`, `plotly.graph_objects`: Interactive visualizations

**Machine Learning & Statistics:**

* `sklearn.ensemble`: RandomForestRegressor, GradientBoostingRegressor
* `sklearn.linear_model`: LinearRegression
* `sklearn.metrics`: MAE, MSE for model evaluation
* `sklearn.model_selection`: train\_test\_split
* `scipy.stats`: Statistical tests & correlation

**Advanced Causal Inference:**

* `pgmpy`: Probabilistic Graphical Models
* `BayesianNetwork`, `TabularCPD`, `VariableElimination`, `BayesianEstimator`

---

### 2. 📊 SalesDataGenerator Class

Generates **synthetic, realistic sales data** with embedded causal relationships.

#### 🔑 Key Features:

* 2 years of weekly data (104 weeks)
* Seasonal patterns + linear trend
* Causal links between competitor actions, discounts, campaigns

#### 🔁 Causal Variables:

* **Exogenous:** `competitor_price`, `market_conditions`, `seasonality`, `trend`
* **Endogenous:** `discount_percent`, `campaign_spend`, `price`, `sales`

#### ⚙️ Mathematical Highlights:

```python
price_elasticity = -1.5
price_effect = (actual_price / base_price) ** price_elasticity
expected_sales = base_sales * seasonality * trend * price_effect * campaign_effect * market_conditions
```

---

### 3. 🔗 CausalGraphBuilder Class

Builds **Bayesian Networks** to model causal structure using `pgmpy`.

#### 📌 Graph Structure:

```python
[
  ('competitor_price_cat', 'discount_cat'),
  ('seasonality_cat', 'discount_cat'),
  ('discount_cat', 'price_cat'),
  ('price_cat', 'sales_cat'),
  ('campaign_cat', 'sales_cat'),
  ('market_cat', 'sales_cat'),
  ('past_sales_cat', 'sales_cat')
]
```

#### 🔎 Methods:

* Bayesian modeling via `VariableElimination`
* Discretization of continuous variables
* Fallback: T-tests + correlation if `pgmpy` unavailable

#### 🌟 Features:

* **Intervention Queries**
* **Counterfactual Reasoning**
* **Confounder Control** via graph structure

---

### 4. 🔮 SalesForecastingModel Class

Implements **ensemble time series forecasting** with causal-aware features.

#### 🔨 Feature Engineering:

* Lagged features: `sales_lag_1`, `discount_lag_4`
* Rolling stats: `sales_ma_8`, `sales_std_12`
* Time-based features: `month`, `quarter`, `week_of_year`
* Interaction terms: `price_discount_interaction`

#### 🧠 Models Used:

* Random Forest
* Gradient Boosting
* Linear Regression
* Final prediction via **ensemble average**

#### 📈 Forecasting Pipeline:

```python
for period in range(n_periods):
    # Apply interventions
    # Update time and lag features
    # Predict sales
    # Append prediction for next iteration
```

---

### 5. 🎲 CounterfactualAnalyzer Class

Performs **"what-if" scenario analysis** to estimate impact of alternative decisions.

#### 🧪 Methodology:

* Forecast Scenario A (actual)
* Forecast Scenario B (counterfactual)
* Compute impact: absolute, relative, total

```python
causal_impact = actual_forecast - counterfactual_forecast
relative_impact = causal_impact / counterfactual_forecast * 100
```

#### 📊 Sensitivity Analysis:

* Sweeps parameter values (e.g., discount from 0% to 30%)
* Visualizes how sales respond to each scenario

---

### 7. 📊 Business Intelligence

#### 📌 KPIs:

* Weekly sales
* Campaign ROI
* Discount effectiveness
* Profit margins

#### 🧠 Automated Insights:

* Price sensitivity (elasticity detection)
* Discount lift calculations
* Campaign ROI analysis
* Seasonality mapping

#### 🎯 Recommendations:

* Dynamic pricing strategies
* Campaign targeting optimization
* Seasonal inventory planning

#### 📤 Export Features:

* Download summary CSV
* Export full dataset

---

## 🧪 Technical Innovations

| Feature            | Innovation                                |
| ------------------ | ----------------------------------------- |
| ✅ Causal Modeling  | Bayesian Networks via pgmpy               |
| 📈 Realistic Data  | Causally driven synthetic data            |
| 🎯 Forecasting     | Multi-model ensemble with causal features |
| 🔍 Counterfactuals | Built-in scenario simulation engine       |
| 📊 Visualization   | Full Plotly integration for interactivity |

---

## 📚 Statistical Foundations

* **Bayesian Graphs**: DAGs, CPDs, conditional independence
* **Time Series**: Lag/rolling stats, intervention-based forecasting
* **ML Ensembles**: Random Forest, Boosting, Model Averaging
* **Stats Tests**: T-tests, correlation, p-values

---

## 🏢 Business Applications

* **💰 Pricing Strategy**: Understand elasticity, optimize price points
* **📣 Campaign Planning**: Maximize ROI and volume
* **📦 Inventory Optimization**: Use demand forecasts
* **⚖️ Risk Assessment**: Compare future outcomes under uncertainty

---

## ✅ Conclusion

* 💡 Realistic synthetic data generation
* 🔗 Causal reasoning & counterfactual simulation
* 🧠 Predictive modeling with causal features
* 📈 Forecasting & strategic planning tools

A powerful demonstration of how **data science can empower business decisions through causal thinking**, not just prediction.

