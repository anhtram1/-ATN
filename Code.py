# Step 1: Setup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import ipywidgets as widgets
from IPython.display import display

# Step 2: Define Local File Paths
file_paths = {
    "Namtex": "NAMTEX.xlsx",
    "Thanh Cong": "CTCP DET MAY - DAU TU - THUONG MAI THANH CONG.xlsx",
    "Det May 29-3": "CTCP DET MAY 29-3.xlsx",
    "Song Hong": "CTCP MAY SONG HONG.xlsx",
    "Viet Tien": "CTCP MAY VIET TIEN.xlsx",
}

# Function to preprocess company data
def preprocess_data(file_path, company_name):
    data = pd.read_excel(file_path, header=1)
    data.columns = data.columns.str.strip()  

    # Financial Ratios (Available for All Companies)
    data['CoS/NR'] = (data['Cost of sales'] / data['Net revenue']) * 100
    data['SE/NR'] = (data['Selling expenses'] / data['Net revenue']) * 100
    data['GAE/NR'] = (data['General and administration expenses'] / data['Net revenue']) * 100
    data['NOP/NR'] = ((data['Gross profit'] + (data['Financial income'] - data['Financial expenses']) - (data['Selling expenses'] + data['General and administration expenses'])) / data['Net revenue']) * 100
    data['Company'] = company_name

    # Extract year from the date column and set it as index
    data['Year'] = pd.to_datetime(data[data.columns[0]]).dt.year # Extract year from date column
    data.dropna(inplace=True)

  # Breakdown of expenses for Namtex
    if company_name == "Namtex":
        data["Selling expenses"] = (
            data["Transportation expenses"] + data["Advertising expenses"] +
            data["Services expenses (selling)"] + data["Staff costs (selling)"] +
            data["Depreciation expenses (selling)"] + data["Other selling expenses"]
        )

        data["General and administration expenses"] = (
            data["Staff costs (G&A)"] + data["Services expenses (G&A)"] +
            data["Outsource expenses (G&A)"] + data["Bank charges"] +
            data["Depreciation expenses (G&A)"] + data["Other G&A expenses"]
        )

        data["Cost of sales"] = (
            data["Finished goods sold and services rendered"] +
            data["Allowance for inventories"]
        )

        data['NPAT/NR'] = ((data['Profit before tax'] - data['Income tax expense - current'] - data['Income tax expense - deferred']) / data['Net revenue']) * 100

    data.set_index('Year', inplace=True)  # Set 'Year' as index # Moved this line to the end

    # Return cleaned dataframe
    return data

# Load and preprocess all companies
company_data = {name: preprocess_data(path, name) for name, path in file_paths.items()}

# Step 3: Correlation Analysis for specific columns
correlation_columns = [
    "CoS/NR", "SE/NR", "GAE/NR", "NOP/NR", "NPAT/NR"]

# Access the preprocessed data for Namtex from the company_data dictionary
data_namtex = company_data["Namtex"]

data_corr = data_namtex[correlation_columns]

# Calculate the correlation matrix
correlation_matrix = data_corr.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    annot_kws={'fontsize': 10}
)

plt.xticks(rotation=15, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Correlation heatmap of financial metrics', fontsize=14, pad=15)
plt.tight_layout()
plt.show()
# Step 5: Regression Analysis

# Regression Model 1: NOP/NR
X_nop = data_namtex[["CoS/NR", "SE/NR", "GAE/NR"]]
y_nop = data_namtex["NOP/NR"]

# Add intercept
X_nop = sm.add_constant(X_nop)

# Build the model
model_nop = sm.OLS(y_nop, X_nop).fit()

# Variance Inflation Factor (VIF)
vif_nop = pd.DataFrame()
vif_nop["Variable"] = X_nop.columns
vif_nop["VIF"] = [variance_inflation_factor(X_nop.values, i) for i in range(X_nop.shape[1])]

# Durbin-Watson statistic
dw_nop = sm.stats.durbin_watson(model_nop.resid)

# Breusch-Pagan test
bp_test_nop = het_breuschpagan(model_nop.resid, X_nop)
bp_test_results_nop = {
    "Lagrange Multiplier Statistic": bp_test_nop[0],
    "p-value": bp_test_nop[1],
    "F-statistic": bp_test_nop[2],
    "F-test p-value": bp_test_nop[3]
}

# Output results
print("\nRegression Model 1: NOP/NR")
print(model_nop.summary())
print("\nVIF for Model 1 (NOP/NR):")
print(vif_nop)
print(f"\nDurbin-Watson Statistic for Model 1 (NOP/NR): {dw_nop}")
print("\nBreusch-Pagan Test for Model 1 (NOP/NR):")
print(bp_test_results_nop)
# Regression Model 2: NPAT/NR
X_npat = data_namtex[["CoS/NR", "SE/NR", "GAE/NR"]]
y_npat = data_namtex["NPAT/NR"]

# Add intercept
X_npat = sm.add_constant(X_npat)

# Build the model
model_npat = sm.OLS(y_npat, X_npat).fit()

# Variance Inflation Factor (VIF)
vif_npat = pd.DataFrame()
vif_npat["Variable"] = X_npat.columns
vif_npat["VIF"] = [variance_inflation_factor(X_npat.values, i) for i in range(X_npat.shape[1])]

# Durbin-Watson statistic
dw_npat = sm.stats.durbin_watson(model_npat.resid)

# Breusch-Pagan test
bp_test_npat = het_breuschpagan(model_npat.resid, X_npat)
bp_test_results_npat = {
    "Lagrange Multiplier Statistic": bp_test_npat[0],
    "p-value": bp_test_npat[1],
    "F-statistic": bp_test_npat[2],
    "F-test p-value": bp_test_npat[3]
}

# Output results
print("\nRegression Model 2: NPAT/NR")
print(model_npat.summary())
print("\nVIF for Model 2 (NPAT/NR):")
print(vif_npat)
print(f"\nDurbin-Watson Statistic for Model 2 (NPAT/NR): {dw_npat}")
print("\nBreusch-Pagan Test for Model 2 (NPAT/NR):")
print(bp_test_results_npat)
# Step 5: Interactive Bar Chart for Comparison
import pandas as pd # Import pandas to work with DataFrames

def plot_comparison(year):
    year_data = all_data[all_data['Year'] == year]

    if year_data.empty:
        print(f"No data available for {year}.")
        return

    ratios = ['CoS/NR', 'SE/NR', 'GAE/NR']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ratio in enumerate(ratios):
        sns.barplot(data=year_data, x='Company', y=ratio, ax=axes[i], palette='viridis')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha='right')
        axes[i].set_ylabel(f'{ratio} (%)')
        axes[i].set_title(f'{ratio} Comparison ({year})')

    plt.tight_layout()
    plt.show()

# Assuming company_data is a dictionary containing DataFrames for each company
# Combine all company data into a single DataFrame called all_data
all_data = pd.concat(company_data.values(), ignore_index=False) # Changed ignore_index to False
all_data = all_data.reset_index() # Reset the index to make 'Year' a column


# Dropdown Menu for Year Selection
year_selector = widgets.Dropdown(
    options=sorted(all_data['Year'].unique()),
    description='Select Year:',
    style={'description_width': 'initial'}
)

# Connect Dropdown to Function
output = widgets.interactive_output(plot_comparison, {'year': year_selector})

# Display Widgets
display(year_selector, output)

from pyngrok import ngrok

# Kill any existing ngrok processes
ngrok.kill()

# Now you can update or start ngrok
ngrok.update()

import os

# Set the ngrok auth token
os.system("ngrok config add-authtoken 2oLXMch4qcGbUpSCADCndGPbLvU_7tTv1xzD5gaX7aZkEXFaR")

# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# --- Preprocessing Function ---
def preprocess_company_data(file, company_name):
    try:
        data = pd.read_excel(file, header=1)
        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]], errors='coerce')
        data['Year'] = data[data.columns[0]].dt.year

        required = [
            'Year', 'Net revenue', 'Cost of sales', 'Finished goods sold and services rendered',
            'Allowance for inventories', 'Selling expenses', 'Transportation expenses',
            'Advertising expenses', 'Services expenses (selling)', 'Staff costs (selling)',
            'Depreciation expenses (selling)', 'Other selling expenses',
            'General and administration expenses', 'Staff costs (G&A)', 'Services expenses (G&A)',
            'Outsource expenses (G&A)', 'Bank charges', 'Depreciation expenses (G&A)',
            'Other G&A expenses', 'Gross profit', 'Financial income', 'Financial expenses'
        ]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        data['CoS/NR'] = (data['Cost of sales'] / data['Net revenue']) * 100
        data['SE/NR'] = (data['Selling expenses'] / data['Net revenue']) * 100
        data['GAE/NR'] = (data['General and administration expenses'] / data['Net revenue']) * 100
        data['NOP/NR'] = (
            (data['Gross profit'] + data['Financial income'] - data['Financial expenses']
             - data['Selling expenses'] - data['General and administration expenses'])
            / data['Net revenue']
        ) * 100
        data['NOP'] = data['Gross profit'] + data['Financial income'] - data['Financial expenses'] - data['Selling expenses'] - data['General and administration expenses']
        data['Company'] = company_name
        return data

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# --- Regression Model ---
def regression_model(data):
    X_nop = data[["CoS/NR", "SE/NR", "GAE/NR"]]
    y_nop = data["NOP"]
    y_nop_nr = data["NOP/NR"]

    X_nop = sm.add_constant(X_nop)

    model_nop = sm.OLS(y_nop, X_nop).fit()
    model_nop_nr = sm.OLS(y_nop_nr, X_nop).fit()

    return {
        "NOP_intercept": model_nop.params["const"],
        "NOP_coef_SE": model_nop.params["SE/NR"],
        "NOP_coef_GAE": model_nop.params["GAE/NR"],
        "NOP_coef_CoS": model_nop.params["CoS/NR"],
        "NOP_NR_intercept": model_nop_nr.params["const"],
        "NOP_NR_coef_SE": model_nop_nr.params["SE/NR"],
        "NOP_NR_coef_GAE": model_nop_nr.params["GAE/NR"],
        "NOP_NR_coef_CoS": model_nop_nr.params["CoS/NR"]
    }

# --- Simulation Function ---
def simulate_expenses_with_regression(data, se_changes, gae_changes, regression_coefficients):
    new_data = data.copy()

    for key in se_changes:
        new_data[key] *= (1 + se_changes[key] / 100)

    new_data['Selling expenses'] = new_data[[
        'Transportation expenses', 'Advertising expenses', 'Services expenses (selling)',
        'Staff costs (selling)', 'Depreciation expenses (selling)', 'Other selling expenses'
    ]].sum(axis=1)

    for key in gae_changes:
        new_data[key] *= (1 + gae_changes[key] / 100)

    new_data['General and administration expenses'] = new_data[[
        'Staff costs (G&A)', 'Services expenses (G&A)', 'Outsource expenses (G&A)',
        'Bank charges', 'Depreciation expenses (G&A)', 'Other G&A expenses'
    ]].sum(axis=1)

    new_data['CoS/NR'] = (new_data['Cost of sales'] / new_data['Net revenue']) * 100
    new_data['SE/NR'] = (new_data['Selling expenses'] / new_data['Net revenue']) * 100
    new_data['GAE/NR'] = (new_data['General and administration expenses'] / new_data['Net revenue']) * 100

    new_data['Net operating profit'] = (
        regression_coefficients["NOP_intercept"] +
        regression_coefficients["NOP_coef_SE"] * new_data['SE/NR'] +
        regression_coefficients["NOP_coef_GAE"] * new_data['GAE/NR'] +
        regression_coefficients["NOP_coef_CoS"] * new_data['CoS/NR']
    )
    new_data['Net operating profit on net revenue'] = (
        regression_coefficients["NOP_NR_intercept"] +
        regression_coefficients["NOP_NR_coef_SE"] * new_data['SE/NR'] +
        regression_coefficients["NOP_NR_coef_GAE"] * new_data['GAE/NR'] +
        regression_coefficients["NOP_NR_coef_CoS"] * new_data['CoS/NR']
    )

    pct_change = pd.DataFrame(index=new_data.index)
    pct_change['Selling expenses'] = ((new_data['Selling expenses'] - data['Selling expenses']) / data['Selling expenses']) * 100
    pct_change['General and administration expenses'] = ((new_data['General and administration expenses'] - data['General and administration expenses']) / data['General and administration expenses']) * 100
    pct_change['Net operating profit'] = ((new_data['Net operating profit'] - data['NOP']) / data['NOP']) * 100
    pct_change['Net operating profit on net revenue'] = ((new_data['Net operating profit on net revenue'] - data['NOP/NR']) / data['NOP/NR']) * 100

    return pct_change

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Financial Dashboard for Namtex", layout="wide")
    st.title("📊 Financial Dashboard for Namtex")

    uploaded_files = st.sidebar.file_uploader("Upload Namtex data files (.xlsx)", accept_multiple_files=True, type=["xlsx"])

    if uploaded_files:
        all_data = []
        for file in uploaded_files:
            company_name = file.name.split(".")[0].upper()
            data = preprocess_company_data(file, company_name)
            if data is not None:
                all_data.append(data)

        if all_data:
            df = pd.concat(all_data)
            namtex_data = df[df['Company'] == 'NAMTEX']
            regression_coefficients = regression_model(namtex_data)

            if not namtex_data.empty:
                selected_year = st.selectbox("📅 Select Year", sorted(namtex_data['Year'].dropna().unique(), reverse=True))
                year_data = namtex_data[namtex_data['Year'] == selected_year].iloc[0]

                st.subheader("💸 Expense Breakdown")
                categories = {
                    "Selling Expenses": [
                        "Transportation expenses", "Advertising expenses", "Services expenses (selling)",
                        "Staff costs (selling)", "Depreciation expenses (selling)", "Other selling expenses"
                    ],
                    "General & Admin Expenses": [
                        "Staff costs (G&A)", "Services expenses (G&A)", "Outsource expenses (G&A)",
                        "Bank charges", "Depreciation expenses (G&A)", "Other G&A expenses"
                    ]
                }
                col1, col2 = st.columns(2)
                for col, (title, cats) in zip([col1, col2], categories.items()):
                    values = [year_data[cat] for cat in cats]
                    fig = go.Figure(data=[go.Pie(labels=cats, values=values, title=title)])
                    col.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 Simulate Expense Changes")
                st.markdown("### 🔧 Selling Expense Adjustments")
                se_changes = {
                    key: st.number_input(f"{key} change (%)", value=0.0)
                    for key in categories["Selling Expenses"]
                }

                st.markdown("### 🏢 G&A Expense Adjustments")
                gae_changes = {
                    key: st.number_input(f"{key} change (%)", value=0.0)
                    for key in categories["General & Admin Expenses"]
                }

                if st.button("Simulate Now"):
                    base_data = namtex_data[namtex_data['Year'] == selected_year].copy()
                    changes_df = simulate_expenses_with_regression(base_data, se_changes, gae_changes, regression_coefficients)
                    st.subheader("📊 Simulation Results (% Change from Original)")
                    st.dataframe(changes_df.round(2))

if __name__ == "__main__":
    main()

import os
import nest_asyncio
from pyngrok import ngrok

# Enable nested asyncio
nest_asyncio.apply()

# Add ngrok token
os.system("ngrok config add-authtoken 2oLXMch4qcGbUpSCADCndGPbLvU_7tTv1xzD5gaX7aZkEXFaR")

# Kill previous ngrok tunnels
ngrok.kill()

# Start Streamlit in the background
os.system("streamlit run app.py &")

# Expose port 8501 via ngrok
# Change port="8501" to port=8501
public_url = ngrok.connect(addr=8501)
print(f"Access your Streamlit app at: {public_url}")
