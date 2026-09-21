"""
HR Attrition & Retention Analytics Dashboard
Internship project

Run:
    pip install -r requirements.txt
    streamlit run hr_attrition_dashboard.py

Dataset:
IBM HR Analytics Employee Attrition & Performance
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="HR Attrition & Retention Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_CANDIDATES = [
    "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    "employee_attrition.csv",
    "WA_Fn-UseC_-HR-Employee-Attrition (1).csv",
]


@st.cache_data
def load_data():
    for filename in DATA_CANDIDATES:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            return df, filename
    return None, None


def prepare_data(df):
    df = df.copy()

    drop_cols = [
        c
        for c in [
            "EmployeeCount",
            "Over18",
            "StandardHours",
            "EmployeeNumber",
        ]
        if c in df.columns
    ]

    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    education_map = {
        1: "Below College",
        2: "College",
        3: "Bachelor",
        4: "Master",
        5: "Doctor",
    }

    satisfaction_map = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High",
    }

    performance_map = {
        1: "Low",
        2: "Good",
        3: "Excellent",
        4: "Outstanding",
    }

    worklife_map = {
        1: "Bad",
        2: "Good",
        3: "Better",
        4: "Best",
    }

    if "Education" in df:
        df["EducationLabel"] = df["Education"].map(education_map)

    if "EnvironmentSatisfaction" in df:
        df["EnvironmentSatisfactionLabel"] = df[
            "EnvironmentSatisfaction"
        ].map(satisfaction_map)

    if "JobInvolvement" in df:
        df["JobInvolvementLabel"] = df["JobInvolvement"].map(
            satisfaction_map
        )

    if "JobSatisfaction" in df:
        df["JobSatisfactionLabel"] = df["JobSatisfaction"].map(
            satisfaction_map
        )

    if "RelationshipSatisfaction" in df:
        df["RelationshipSatisfactionLabel"] = df[
            "RelationshipSatisfaction"
        ].map(satisfaction_map)

    if "PerformanceRating" in df:
        df["PerformanceRatingLabel"] = df["PerformanceRating"].map(
            performance_map
        )

    if "WorkLifeBalance" in df:
        df["WorkLifeBalanceLabel"] = df["WorkLifeBalance"].map(
            worklife_map
        )

    df["AttritionFlag"] = (
        df["Attrition"].eq("Yes")
    ).astype(int)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=[
            "18–25",
            "26–35",
            "36–45",
            "46–55",
            "56+",
        ],
        include_lowest=True,
    )

    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 1, 3, 5, 10, 100],
        labels=[
            "0–1 yr",
            "2–3 yrs",
            "4–5 yrs",
            "6–10 yrs",
            "10+ yrs",
        ],
        include_lowest=True,
    )

    df["IncomeGroup"] = pd.cut(
        df["MonthlyIncome"],
        bins=[-1, 3000, 5000, 8000, 12000, np.inf],
        labels=[
            "< 3k",
            "3k–5k",
            "5k–8k",
            "8k–12k",
            "12k+",
        ],
        include_lowest=True,
    )

    return df


def rate_table(df, column):
    g = (
        df.groupby(column, observed=False)
        .agg(
            Employees=("AttritionFlag", "size"),
            Attritions=("AttritionFlag", "sum"),
        )
        .reset_index()
    )

    g["AttritionRate"] = (
        100 * g["Attritions"] / g["Employees"]
    )

    return g.sort_values(
        "AttritionRate",
        ascending=False,
    )


def build_model(df):
    target = "AttritionFlag"

    features = [
        "Age",
        "BusinessTravel",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "Gender",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    features = [
        c for c in features
        if c in df.columns
    ]

    X = df[features].copy()
    y = df[target]

    numeric = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical = [
        c for c in features
        if c not in numeric
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    model = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            pred,
        ),
        "Precision": precision_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            prob,
        ),
    }

    feature_names = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    coefficients = (
        model
        .named_steps["classifier"]
        .coef_[0]
    )

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
        }
    )

    coef_df["AbsCoefficient"] = (
        coef_df["Coefficient"].abs()
    )

    coef_df = (
        coef_df
        .sort_values(
            "AbsCoefficient",
            ascending=False,
        )
        .head(15)
    )

    return model, metrics, coef_df


df, filename = load_data()

st.title(
    "📊 HR Attrition & Retention Analytics"
)

st.caption(
    "Business Intelligence dashboard: "
    "facts → insights → risks/opportunities → actions"
)

if df is None:
    st.error("Dataset not found.")

    st.markdown(
        "Download the CSV from the dataset link "
        "in README.md and place it in the same "
        "folder as `hr_attrition_dashboard.py`."
    )

    st.stop()

df = prepare_data(df)

# Sidebar
st.sidebar.header("Filters")

departments = [
    "All"
] + sorted(
    df["Department"]
    .dropna()
    .unique()
    .tolist()
)

selected_dept = st.sidebar.selectbox(
    "Department",
    departments,
)

genders = [
    "All"
] + sorted(
    df["Gender"]
    .dropna()
    .unique()
    .tolist()
)

selected_gender = st.sidebar.selectbox(
    "Gender",
    genders,
)

filtered = df.copy()

if selected_dept != "All":
    filtered = filtered[
        filtered["Department"]
        == selected_dept
    ]

if selected_gender != "All":
    filtered = filtered[
        filtered["Gender"]
        == selected_gender
    ]

# KPIs
total = len(filtered)

attritions = int(
    filtered["AttritionFlag"].sum()
)

attrition_rate = (
    attritions / total * 100
    if total
    else 0
)

avg_income = (
    filtered["MonthlyIncome"].mean()
    if total
    else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Employees",
    f"{total:,}",
)

c2.metric(
    "Attrition Count",
    f"{attritions:,}",
)

c3.metric(
    "Attrition Rate",
    f"{attrition_rate:.1f}%",
)

c4.metric(
    "Avg Monthly Income",
    f"${avg_income:,.0f}",
)

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "Executive Overview",
        "Drivers & Risk",
        "Prediction Model",
    ]
)

# ------------------------------------------------
# EXECUTIVE OVERVIEW
# ------------------------------------------------

with tab1:

    st.subheader(
        "Executive Overview"
    )

    col1, col2 = st.columns(2)

    with col1:

        dept = rate_table(
            filtered,
            "Department",
        )

        dept_sorted = dept.sort_values(
            "AttritionRate"
        )

        fig = px.bar(
            dept_sorted,
            x="AttritionRate",
            y="Department",
            orientation="h",
            text=dept_sorted[
                "AttritionRate"
            ].round(1),
            title="Attrition Rate by Department",
            labels={
                "AttritionRate":
                    "Attrition Rate (%)"
            },
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        age = rate_table(
            filtered,
            "AgeGroup",
        )

        age_sorted = age.sort_index()

        fig = px.bar(
            age_sorted,
            x="AgeGroup",
            y="AttritionRate",
            text=age_sorted[
                "AttritionRate"
            ].round(1),
            title="Attrition Rate by Age Group",
            labels={
                "AttritionRate":
                    "Attrition Rate (%)"
            },
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "Business Interpretation"
    )

    st.write(
        "Use the dashboard to identify where "
        "attrition is concentrated, then drill "
        "into job role, overtime, satisfaction, "
        "tenure and compensation. The objective "
        "is not to create more charts; it is to "
        "identify a small number of actionable "
        "retention priorities."
    )

    role = (
        rate_table(
            filtered,
            "JobRole",
        )
        .head(10)
    )

    role_sorted = role.sort_values(
        "AttritionRate"
    )

    fig = px.bar(
        role_sorted,
        x="AttritionRate",
        y="JobRole",
        orientation="h",
        title="Highest Attrition Job Roles",
        labels={
            "AttritionRate":
                "Attrition Rate (%)"
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ------------------------------------------------
# DRIVERS AND RISK
# ------------------------------------------------

with tab2:

    st.subheader(
        "Drivers and Risk Indicators"
    )

    a, b, c = st.columns(3)

    overtime = (
        filtered
        .groupby("OverTime")[
            "AttritionFlag"
        ]
        .mean()
        * 100
    )

    ot_yes = float(
        overtime.get(
            "Yes",
            np.nan,
        )
    )

    ot_no = float(
        overtime.get(
            "No",
            np.nan,
        )
    )

    a.metric(
        "Overtime Attrition",
        f"{ot_yes:.1f}%",
    )

    b.metric(
        "Non-Overtime Attrition",
        f"{ot_no:.1f}%",
    )

    b_delta = (
        ot_yes - ot_no
        if not np.isnan(ot_yes)
        and not np.isnan(ot_no)
        else np.nan
    )

    c.metric(
        "Overtime Gap",
        (
            f"{b_delta:.1f} pp"
            if not np.isnan(b_delta)
            else "N/A"
        ),
    )

    col1, col2 = st.columns(2)

    with col1:

        overtime_df = (
            filtered
            .groupby("OverTime", as_index=False)[
                "AttritionFlag"
            ]
            .mean()
        )

        overtime_df[
            "AttritionRate"
        ] = (
            overtime_df["AttritionFlag"]
            * 100
        )

        fig = px.bar(
            overtime_df,
            x="OverTime",
            y="AttritionRate",
            text="AttritionRate",
            title="Overtime and Attrition",
            labels={
                "AttritionRate":
                    "Attrition Rate (%)"
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        fig = px.box(
            filtered,
            x="Attrition",
            y="MonthlyIncome",
            title=(
                "Monthly Income Distribution "
                "by Attrition"
            ),
            labels={
                "MonthlyIncome":
                    "Monthly Income"
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "Risk / Opportunity / Action"
    )

    st.markdown(
        """
**Risk:** High attrition in a department or job role can create replacement, training and productivity costs.

**Opportunity:** Segments with lower attrition can be compared with high-risk segments to identify practices associated with stronger retention.

**Action:** Prioritize retention reviews for high-risk segments, especially where overtime, low tenure, compensation or satisfaction indicators are simultaneously elevated.

**Important:** These are associations in a synthetic cross-sectional dataset, not proof that any single factor causes an employee to leave.
"""
    )


# ------------------------------------------------
# PREDICTION MODEL
# ------------------------------------------------

with tab3:

    st.subheader(
        "Optional Attrition Risk Model"
    )

    st.caption(
        "Logistic regression is included as "
        "supporting analytics; dashboard "
        "insights remain the primary deliverable."
    )

    if st.button(
        "Train / Refresh Model"
    ):

        with st.spinner(
            "Training model..."
        ):

            model, metrics, coef_df = (
                build_model(df)
            )

        st.session_state["model"] = model
        st.session_state["metrics"] = metrics
        st.session_state["coef_df"] = coef_df

    if "metrics" not in st.session_state:

        st.info(
            "Click 'Train / Refresh Model' "
            "to calculate model metrics."
        )

    else:

        metrics = (
            st.session_state["metrics"]
        )

        cols = st.columns(5)

        for col, (
            name,
            value,
        ) in zip(
            cols,
            metrics.items(),
        ):

            col.metric(
                name,
                f"{value:.3f}",
            )

        st.subheader(
            "Model Signals"
        )

        st.dataframe(
            st.session_state[
                "coef_df"
            ][
                [
                    "Feature",
                    "Coefficient",
                ]
            ].reset_index(
                drop=True
            ),
            use_container_width=True,
        )

        st.warning(
            "Do not use this model as an "
            "automated hiring, firing or "
            "promotion decision system. "
            "It is a learning/analytics aid "
            "and should be reviewed for bias, "
            "fairness and context."
        )


st.divider()

st.caption(
    f"Data file: {filename} | "
    "Dataset source: IBM HR Analytics "
    "Employee Attrition & Performance "
    "(Kaggle). This dashboard is for "
    "educational analytics and "
    "decision-support purposes."
)