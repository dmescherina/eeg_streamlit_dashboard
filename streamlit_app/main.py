# main.py
# Title: SHAP values explorer

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import config

st.set_page_config(page_title="SHAP values explorer", layout="wide")

st.title("Interactive exploration of SHAP values")
st.write("Navigate to the pages in the sidebar for other visualizations")

# Load dataset based on toggle
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data_source = st.radio(
    "Select Data Source",
    options=["Averaged SHAP Values", "Summed SHAP Values"],
    index=0
)

# Set file path and aggregation method based on selection
if data_source == "Averaged SHAP Values":
    file_path = config.DATA_FILE_SHAP_MEAN  # Path to averaged SHAP values
    aggregation_method = "mean"          # Use mean for aggregation
else:
    file_path = config.DATA_FILE_SHAP_SUM  # Path to summed SHAP values
    aggregation_method = "sum"           # Use sum for aggregation

# Load the selected dataset
df = load_data(file_path)

# Sidebar Filters
st.sidebar.header("Filters")
selected_clusters = st.sidebar.multiselect("Select Clusters", df['Cluster'].unique(), default=df['Cluster'].unique())
selected_time_periods = st.sidebar.multiselect("Select Time Periods", df['Time Period'].unique(), default=df['Time Period'].unique())
selected_frequencies = st.sidebar.multiselect("Select Frequencies", df['Frequency'].unique(), default=df['Frequency'].unique())
selected_stimuli = st.sidebar.multiselect("Select Stimuli", df['Stimulus'].unique(), default=df['Stimulus'].unique())
selected_scores = st.sidebar.multiselect("Select Target Scores", df['TargetScore'].unique(), default=df['TargetScore'].unique())

# Filter the DataFrame based on selections
filtered_df = df[
    (df['Cluster'].isin(selected_clusters)) &
    (df['Time Period'].isin(selected_time_periods)) &
    (df['Frequency'].isin(selected_frequencies)) &
    (df['Stimulus'].isin(selected_stimuli)) &
    (df['TargetScore'].isin(selected_scores))
]

# Bar Plot
st.header("Bar Plot of SHAP Values")
bar_x_axis = st.selectbox("Select X-Axis for Bar Plot", ["Cluster", "Frequency", "Time Period"])
bar_row_partition = st.selectbox("Row Partition by (Optional)", [None, "Stimulus", "TargetScore", "Cluster", "Frequency", "Time Period"])
bar_col_partition = st.selectbox("Column Partition by (Optional)", [None, "Stimulus", "TargetScore", "Cluster", "Frequency", "Time Period"])

# Prepare group-by columns
groupby_columns = [bar_x_axis]
if bar_row_partition:
    groupby_columns.append(bar_row_partition)
if bar_col_partition:
    groupby_columns.append(bar_col_partition)

# Aggregate SHAP values for the bar plot
bar_data = filtered_df.groupby(groupby_columns).agg({"SHAP_value": "sum"}).reset_index()

# Create Bar Plot
fig_bar = px.bar(
    bar_data,
    x=bar_x_axis,
    y="SHAP_value",
    color=bar_col_partition,
    facet_row=bar_row_partition,
    barmode="group",
    labels={"SHAP_value": f"{aggregation_method.capitalize()} SHAP Value"},
    title=f"SHAP Values by {bar_x_axis} ({data_source})",
    height=800,
    width=1200,
)
st.plotly_chart(fig_bar, use_container_width=True)

# Box Plot
st.header("Box Plot of SHAP Values")
box_x_axis = st.selectbox("Select X-Axis for Box Plot", ["Cluster", "Frequency", "Time Period", "Stimulus", "TargetScore"])
box_y_axis = st.selectbox("Select Y-Axis for Box Plot", ["SHAP_value"])
box_row_partition = st.selectbox("Row Partition for Box Plot (Optional)", [None, "Stimulus", "TargetScore", "Cluster", "Frequency", "Time Period"])
box_col_partition = st.selectbox("Column Partition for Box Plot (Optional)", [None, "Stimulus", "TargetScore", "Cluster", "Frequency", "Time Period"])

# Create Box Plot
fig_box = px.box(
    filtered_df,
    x=box_x_axis,
    y=box_y_axis,
    color=box_col_partition,
    facet_row=box_row_partition,
    labels={"SHAP_value": f"{aggregation_method.capitalize()} SHAP Value"},
    title=f"Box Plot of SHAP Values ({box_x_axis} vs {box_y_axis}) ({data_source})",
    height=800,
    width=1200,
)
st.plotly_chart(fig_box, use_container_width=True)