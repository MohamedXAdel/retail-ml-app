# Online Retail Customer Segmentation ML Web App

**Live Demo**: [https://retail-ml-app-production.up.railway.app/predict]

---

## Project Overview

This is a **full end-to-end Machine Learning web application** that:
- Analyzes the **UCI Online Retail Dataset**
- Performs **RFM (Recency, Frequency, Monetary)** customer segmentation
- Uses **K-Means Clustering** to group customers
- Deploys a **live interactive web app** using **FastAPI + HTML/CSS**

---

## Tech Stack

| Component         | Technology Used                     |
|-------------------|-------------------------------------|
| Data Processing   | `pandas`, `numpy`                   |
| ML Model          | `scikit-learn` (K-Means)            |
| Model Saving      | `joblib`                            |
| Backend           | **FastAPI**                         |
| Frontend          | HTML, CSS, Jinja2                   |
| Server            | `uvicorn`                           |
| Deployment        | **Railway.app** (Free, No Credit Card) |
| Version Control   | Git + GitHub Desktop                |

---

## Dataset

- **Source**: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- **Size**: 541,909 transactions
- **Features Used**:
  - `InvoiceDate` → Recency
  - `InvoiceNo` → Frequency
  - `Quantity × UnitPrice` → Monetary

---

## ML Pipeline (Step-by-Step)

1. **Data Cleaning**
   - Dropped missing `CustomerID`
   - Removed negative/zero `Quantity` or `UnitPrice`
   - Removed duplicates

2. **RFM Feature Engineering**
   ```python
   Recency   = Days since last purchase
   Frequency = Number of unique invoices
   Monetary  = Total spent (Quantity × UnitPrice)
