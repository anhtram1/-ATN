# 📊 Financial Dashboard for Namtex

A Streamlit-based financial dashboard to visualize and simulate expense changes for Namtex and other companies using regression models and interactive charts.

---

## 🚀 Features
- Upload Namtex's Excel financial data file
- Preprocess and visualize financial metrics
- Interactive pie charts for Selling & G&A expenses
- Simulate expense changes and predict Net Operating Profit (NOP)
- Regression analysis (via statsmodels)

---

## 🏗️ Project Structure
```
project-root/
├── app.py                  # Main Streamlit dashboard
├── Requirements.txt        # Dependencies
├── README.md               # Project overview
├── company_data/           # Folder for uploaded Excel files
└── Code.py        
```

---

## 🛠️ Installation
```bash
# Clone this repository
$ git clone https://github.com/anhtram1/-ATN 
$ cd namtex-dashboard

# Create virtual environment (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r Requirements.txt
```

---

## ▶️ Run the App
```bash
streamlit run app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

---

## 📁 Sample Data Structure
Ensure your `.xlsx` files have the required columns with a header on the **second row** (index 1):

- `Net revenue`
- `Cost of sales`
- `Selling expenses`, `Transportation expenses`, ...
- `General and administration expenses`, `Staff costs (G&A)`, ...
- `Gross profit`, `Financial income`, `Financial expenses`

---



