# pages/feature_comparison.py
# Title: Comparison of Linear and LightGBM parameters

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import config

# Set Streamlit page layout to wide
st.set_page_config(page_title="Model Parameters Comparison", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_FILE_COEF)

df_all_test = load_data()

# Streamlit App Title
st.title("Comparison of Coefficients from Two Models")

# Sidebar Filters
st.sidebar.header("Filters")
selected_clusters = st.sidebar.multiselect("Select Clusters", df_all_test['Cluster'].unique(), default=df_all_test['Cluster'].unique())
selected_time_periods = st.sidebar.multiselect("Select Time Periods", df_all_test['Time Period'].unique(), default=df_all_test['Time Period'].unique())
selected_frequencies = st.sidebar.multiselect("Select Frequencies", df_all_test['Frequency'].unique(), default=df_all_test['Frequency'].unique())
selected_conditions = st.sidebar.multiselect("Select Stimulus", df_all_test['Stimulus'].unique(), default=df_all_test['Stimulus'].unique())
selected_scores = st.sidebar.multiselect("Select Scores", df_all_test['TargetScore'].unique(), default=df_all_test['TargetScore'].unique())

# Toggle: Raw vs Scaled Values
show_scaled = st.checkbox("Show Scaled Values (-1 to 1)", value=True)

# New toggle: Partition Stimulus and TargetScore
partition_rows = st.checkbox("Partition by Stimulus (Rows)", value=False)
partition_cols = st.checkbox("Partition by TargetScore (Columns)", value=False)

# Filter DataFrame based on selection
filtered_df = df_all_test[
    (df_all_test['Cluster'].isin(selected_clusters)) &
    (df_all_test['Time Period'].isin(selected_time_periods)) &
    (df_all_test['Frequency'].isin(selected_frequencies)) &
    (df_all_test['Stimulus'].isin(selected_conditions)) &
    (df_all_test['TargetScore'].isin(selected_scores))
]

# Aggregate (Average) coefficients by Feature, Stimulus, and TargetScore
grouped_df = filtered_df.groupby(["features", "Stimulus", "TargetScore"], as_index=False).agg({
    "params_significant": "mean",
    "SHAP_value": "mean"
})

# Apply Min-Max Scaling after averaging
if show_scaled:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    grouped_df[['params_significant', 'SHAP_value']] = scaler.fit_transform(
        grouped_df[['params_significant', 'SHAP_value']]
    )

# Melt the DataFrame to stack Linear Model and SHAP values for easier plotting
melted_df = grouped_df.melt(
    id_vars=["features", "Stimulus", "TargetScore"],
    value_vars=["params_significant", "SHAP_value"],
    var_name="Model",
    value_name="Coefficient Value"
)

# Map readable labels for models
melted_df["Model"] = melted_df["Model"].map({
    "params_significant": "Linear Model Coefficients",
    "SHAP_value": "LightGBM SHAP Values"
})

# Select row and column facets dynamically
facet_row = "Stimulus" if partition_rows else None
facet_col = "TargetScore" if partition_cols else None

# Create Plotly Express Figure
fig = px.bar(
    melted_df,
    x="features",
    y="Coefficient Value",
    color="Model",  # Linear vs SHAP
    barmode="group",
    title=f"Comparison of {'Scaled' if show_scaled else 'Raw'} Coefficients from Two Models",
    facet_row=facet_row,
    facet_col=facet_col,
    labels={"Coefficient Value": "Coefficient Value", "features": "Features"},
    height=900
)

# Rotate x-axis labels in all subplots
fig.update_xaxes(tickangle=45)

# Update layout for readability
fig.update_layout(
    title_font_size=20,
    xaxis=dict(
        title="Features",
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        title="Coefficient Values" + (" (-1 to 1)" if show_scaled else " (Raw)"),
        title_font_size=18,
        tickfont=dict(size=14),
        showgrid=True
    ),
    barmode="relative",  # Side-by-side comparison for Linear vs SHAP
)

# Display Plotly Figure in Streamlit with full width
st.plotly_chart(fig, use_container_width=True)