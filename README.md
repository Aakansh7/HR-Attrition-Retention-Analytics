# HR Attrition & Retention Analytics

## Turning Workforce Data into Actionable Insights

## 1. Project Overview

This project converts employee-level HR data into a Business Intelligence dashboard for analyzing employee attrition and retention.

The project focuses on:

- KPI monitoring
- Attrition analysis
- Department and job-role analysis
- Overtime analysis
- Tenure analysis
- Compensation analysis
- Workforce risk indicators
- Optional attrition-risk modelling
- Translating findings into risks, opportunities and recommended actions

The project follows the Business Intelligence framework:

**Data → Information → Insight → Decision → Action**

---

## 2. Problem Statement

Employee turnover can create recruitment, onboarding and productivity costs.

HR teams need a concise way to identify where employee attrition is concentrated and which workforce characteristics should receive further investigation.

The project addresses the following questions:

1. What is the overall attrition level?
2. Which departments and job roles show higher attrition rates?
3. How are overtime, tenure, satisfaction and compensation associated with attrition?
4. Which employee segments deserve additional retention review?
5. How can the analysis be converted into practical HR actions?

---

## 3. Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

**Dataset source:**

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset contains:

- 1,470 employee records
- 35 columns

The dataset is a fictional/synthetic dataset created for analytics and modelling purposes.

### Dataset Setup

Download the CSV dataset and place it in the same folder as:

`hr_attrition_dashboard.py`

The application accepts the following filenames:

- `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- `employee_attrition.csv`

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn
- Logistic Regression

---

## 5. Dashboard Structure

### Executive Overview

The Executive Overview provides:

- Total employees
- Attrition count
- Attrition rate
- Average monthly income
- Attrition by department
- Attrition by age group
- Attrition by job role

Interactive filters are provided for:

- Department
- Gender

---

### Drivers & Risk

This section analyses:

- Overtime vs attrition
- Monthly income distribution
- Risk indicators
- Opportunities for further investigation
- Recommended actions

The objective is to identify workforce segments that may require additional retention review.

---

### Prediction Model

An optional Logistic Regression model is included as supporting analytics.

The model reports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Model coefficient signals

The model is intended as a learning and decision-support component and not as an automated employment decision system.

---

## 6. Business Intelligence Logic

The project follows this chain:

### Fact

What is happening in the workforce data?

Examples:

- Number of employees
- Number of attritions
- Attrition rate
- Workforce characteristics

### Insight

Which department, job role or workforce factor is associated with higher attrition?

### Risk

Where could continued attrition create workforce pressure?

### Opportunity

Which lower-risk segments or practices can be studied for retention ideas?

### Action

What should HR investigate or pilot next?

The project treats observed relationships as associations and does not claim that correlation proves causation.

---

## 7. Key Dataset-Level Findings

The dataset contains:

- 1,470 employee records
- 237 records labelled as Attrition = Yes
- Overall attrition rate of approximately 16.1%

Department-level employee counts include:

- Research & Development: 961
- Sales: 446
- Human Resources: 63

The dashboard calculates attrition rates dynamically from the dataset and allows users to filter the analysis.

---

## 8. How to Run the Project

### Step 1 — Install dependencies

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt