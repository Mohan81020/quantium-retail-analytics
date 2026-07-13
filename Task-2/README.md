# 🛒 Quantium Retail Analytics – Trial Store Analysis

## 📌 Project Overview

This project analyzes the impact of a retail trial conducted by Quantium.

The objective is to identify the most suitable control store for each trial store using historical sales performance and then evaluate whether the trial significantly improved business performance.

The complete analysis is performed using **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **SciPy**.

---

## 🎯 Business Objective

Three stores were selected for a retail trial:

- Store 77
- Store 86
- Store 88

The project aims to:

- Find the most similar control store before the trial.
- Compare trial and control store performance.
- Scale control store sales.
- Measure percentage uplift.
- Perform statistical significance testing.
- Determine whether the trial was successful.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook

---



## 📊 Analysis Workflow

### Step 1

Load and clean transaction data.

### Step 2

Create monthly KPIs for every store.

- Total Sales
- Number of Customers
- Average Transactions per Customer

### Step 3

Define the pre-trial period.

### Step 4

Calculate Pearson Correlation for:

- Sales
- Customers
- Average Transactions per Customer

### Step 5

Calculate Magnitude Similarity for each KPI.

### Step 6

Combine correlation and magnitude scores.

### Step 7

Compute Final Similarity Score.

### Step 8

Identify the best control store for each trial store.

### Step 9

Compare monthly performance.

### Step 10

Scale control store metrics using the pre-trial period.

### Step 11

Calculate percentage difference during the trial period.

### Step 12

Perform statistical significance testing using a t-test.

### Step 13

Visualize results and summarize findings.

---

## 📈 Visualizations

The project includes visualizations for:

- Top Similar Stores
- Monthly Sales Comparison
- Monthly Customer Comparison
- Monthly Transactions per Customer
- Scaled Control Store Comparison
- Confidence Interval Analysis
- Trial Performance Comparison

---

## 📋 Results Summary

The analysis identifies the best matching control store for each trial store and evaluates whether the trial generated statistically significant improvements in business performance.

---

## 📁 Report

A detailed project report is available here:

➡️ **Task-2 Report**



## 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/Mohan81020/quantium-retail-analytics.git
```

Open Jupyter Notebook

```bash
jupyter notebook
```

Navigate to

```
Task-2/Notebook/
```

Run all cells.

---

## 👨‍💻 Author

**Mohan Kumar**

GitHub:
https://github.com/Mohan81020
