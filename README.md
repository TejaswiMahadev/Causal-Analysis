# 📉 Causal Impact Analysis of Discounts on Retail Sales

This project investigates the **causal effect of discount and marketing campaigns** on weekly sales using real-world retail data. It combines **causal inference**, **machine learning forecasting**, and **counterfactual simulation** to guide better pricing and promotional decisions.

---

## 🧠 What This Project Does

- ⏳ Aggregates transactional data into weekly sales
- 🔍 Models causal relationships using Bayesian networks
- 📊 Forecasts sales using ensemble ML models
- 🔄 Simulates counterfactual scenarios: "What if no discount was offered?"
- 📈 Performs sensitivity analysis to understand impact ranges
- 📝 Generates business insights and visual summaries

---

## 📦 Tech Stack

| Area             | Libraries Used                             |
|------------------|--------------------------------------------|
| Data Processing  | `pandas`, `numpy`, `matplotlib`, `seaborn` |
| Causal Inference | `pgmpy`, `scipy.stats`                     |
| Forecasting      | `scikit-learn`, `statsmodels`              |
| Counterfactuals  | Custom modeling logic                      |
| Visualization    | `matplotlib`, `seaborn`                    |

---

## 📈 Example Use Cases

- 📦 *"How much did a 20% discount impact sales last month?"*
- 📊 *"Would our revenue be higher without running this campaign?"*
- 🔮 *"What happens to profit if we change campaign spend from $500 to $1000?"*

---
## 🧪 Dataset Info

- This project uses the UCI Online Retail Dataset, which contains transactional data from a UK-based online store from 2010-2011.
- We group this data into weekly sales and simulate discounts/campaigns for causal testing.

## ✅ We get 
- 📊 Sales over time and by discount level
- 📉 Causal effect of discounts on sales
- 🔄 Counterfactual forecasts with and without interventions
- 📦 Sensitivity analysis of sales based on discount or campaign levels

