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
    "Namtex": "CT TNHH DET NHUOM NAM PHUONG.xlsx",
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
# Add 'Year' column if it's not already present
# all_data['Year'] = all_data['Year'].astype(int) # Convert 'Year' column to integers for proper sorting

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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# Function to preprocess company data
def preprocess_company_data(file, company_name):
    try:
        data = pd.read_excel(file, header=1)

        # Extract date and convert to year
        first_col_name = data.columns[0]
        data[first_col_name] = pd.to_datetime(data[first_col_name], errors='coerce')
        data['Year'] = data[first_col_name].dt.year

        # Required columns & expense breakdown
        required_columns = [
            'Year', 'Net revenue', 
            'Cost of sales', 'Finished goods sold and services rendered', 'Allowance for inventories',
            'Selling expenses', 'Transportation expenses', 'Advertising expenses', 'Services expenses (selling)',
            'Staff costs (selling)', 'Depreciation expenses (selling)', 'Other selling expenses',
            'General and administration expenses', 'Staff costs (G&A)', 'Services expenses (G&A)',
            'Outsource expenses (G&A)', 'Bank charges', 'Depreciation expenses (G&A)', 'Other G&A expenses',
            'Gross profit', 'Financial income', 'Financial expenses'
        ]

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Compute % values relative to Net Revenue
        data['CoS/NR'] = (data['Cost of sales'] / data['Net revenue']) * 100
        data['SE/NR'] = (data['Selling expenses'] / data['Net revenue']) * 100
        data['GAE/NR'] = (data['General and administration expenses'] / data['Net revenue']) * 100
        data['NOP/NR'] = ((data['Gross profit'] + (data['Financial income'] - data['Financial expenses']) -
                          (data['Selling expenses'] + data['General and administration expenses'])) /
                          data['Net revenue']) * 100

        data['Company'] = company_name
        return data
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Function to calculate new expenses based on % changes
def simulate_expenses(data, se_changes, gae_changes):
    new_data = data.copy()

    # Adjust Selling Expenses breakdown
    for key in se_changes:
        new_data[key] *= (1 + se_changes[key] / 100)
    new_data['Selling expenses'] = new_data[['Transportation expenses', 'Advertising expenses', 
                                             'Services expenses (selling)', 'Staff costs (selling)', 
                                             'Depreciation expenses (selling)', 'Other selling expenses']].sum(axis=1)

    # Adjust GAE breakdown
    for key in gae_changes:
        new_data[key] *= (1 + gae_changes[key] / 100)
    new_data['General and administration expenses'] = new_data[['Staff costs (G&A)', 'Services expenses (G&A)', 
                                                                'Outsource expenses (G&A)', 'Bank charges', 
                                                                'Depreciation expenses (G&A)', 'Other G&A expenses']].sum(axis=1)

    # Update Net Operating Profit
    new_data['Net operating profit'] = (new_data['Gross profit'] + (new_data['Financial income'] - new_data['Financial expenses']) -
                           (new_data['Selling expenses'] + new_data['General and administration expenses']))

    # Calculate percentage change
    percentage_changes = pd.DataFrame()
    for col in ['Selling expenses', 'General and administration expenses', 'Net operating profit']:
        percentage_changes[col] = ((new_data[col] - data[col]) / data[col]) * 100

    return percentage_changes

def main():
    st.set_page_config(page_title="Financial Dashboard", layout="wide")
    st.title("📊 Financial Dashboard")

    uploaded_files = st.sidebar.file_uploader("Upload Excel files", accept_multiple_files=True, type=['xlsx'])

    if uploaded_files:
        all_data = []
        for file in uploaded_files:
            company_name = file.name.split(".")[0]
            data = preprocess_company_data(file, company_name)
            if data is not None:
                all_data.append(data)

        if all_data:
            df = pd.concat(all_data)
            namtex_data = df[df['Company'] == 'CT TNHH DET NHUOM NAM PHUONG']

            if not namtex_data.empty:
                selected_year = st.selectbox("📅 Select Year", sorted(namtex_data['Year'].dropna().unique(), reverse=True))
                year_data = namtex_data[namtex_data['Year'] == selected_year].iloc[0]

                # Pie Charts: Expense Breakdown
                expense_categories = {
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
                for col, (title, categories) in zip([col1, col2], expense_categories.items()):
                    labels = categories
                    values = [year_data[cat] for cat in categories]
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, title=title)])
                    col.plotly_chart(fig)

                # User input for simulation
                st.subheader("📈 Simulate Changes")
                se_changes = {key: st.number_input(f"{key} Change (%)", value=0.0) for key in [
                    "Transportation expenses", "Advertising expenses", "Services expenses (selling)", 
                    "Staff costs (selling)", "Depreciation expenses (selling)", "Other selling expenses"
                ]}
                gae_changes = {key: st.number_input(f"{key} Change (%)", value=0.0) for key in [
                    "Staff costs (G&A)", "Services expenses (G&A)", "Outsource expenses (G&A)", 
                    "Bank charges", "Depreciation expenses (G&A)", "Other G&A expenses"
                ]}

                if st.button("Run Simulation"):
                    simulated_data = simulate_expenses(namtex_data, se_changes, gae_changes)
                    st.subheader("📊 Simulation Results (Percentage Change)")
                    st.write(simulated_data)

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
