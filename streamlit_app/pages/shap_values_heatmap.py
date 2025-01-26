# pages/shap_values_heatmap.py
#Title: SHAP Values Heatmap

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
import config

# Set Streamlit page layout to wide
st.set_page_config(page_title="Hierarchical SHAP Heatmap", layout="wide")

## Load dataset based on toggle
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

# Streamlit App Title
st.title("Hierarchical SHAP Heatmap")

# Sidebar Filters
st.sidebar.header("Filters")
selected_clusters = st.sidebar.multiselect("Select Clusters", df['Cluster'].unique(), default=df['Cluster'].unique())
selected_time_periods = st.sidebar.multiselect("Select Time Periods", df['Time Period'].unique(), default=df['Time Period'].unique())
selected_frequencies = st.sidebar.multiselect("Select Frequencies", df['Frequency'].unique(), default=df['Frequency'].unique())
selected_stimuli = st.sidebar.multiselect("Select Stimulus", df['Stimulus'].unique(), default=df['Stimulus'].unique())
selected_scores = st.sidebar.multiselect("Select Target Scores", df['TargetScore'].unique(), default=df['TargetScore'].unique())

# Filter the DataFrame based on selections
filtered_df = df[
    (df['Cluster'].isin(selected_clusters)) &
    (df['Time Period'].isin(selected_time_periods)) &
    (df['Frequency'].isin(selected_frequencies)) &
    (df['Stimulus'].isin(selected_stimuli)) &
    (df['TargetScore'].isin(selected_scores))
]

# Create a pivot table for the heatmap (aggregating SHAP values)
pivot_df = filtered_df.pivot_table(
    index=["Time Period", "Frequency"],  # Use Time Period and Frequency as hierarchical row labels
    columns="Cluster",  # Use Clusters as columns
    values="SHAP_value",
    aggfunc=aggregation_method  # Aggregate SHAP values by dynamic aggregation method
)

# Convert the pivot table to a matrix for the heatmap
heatmap_matrix = pivot_df.values
row_labels = [f"{time}, {freq}" for time, freq in pivot_df.index]  # Combined labels for rows
col_labels = pivot_df.columns.tolist()  # Cluster names

# Apply hierarchical clustering to sort rows and columns
row_linkage = linkage(heatmap_matrix, method="ward")
col_linkage = linkage(heatmap_matrix.T, method="ward")

# Get the new order of rows and columns
sorted_row_idx = leaves_list(row_linkage)
sorted_col_idx = leaves_list(col_linkage)

# Reorder the matrix, row labels, and column labels
heatmap_matrix = heatmap_matrix[sorted_row_idx, :]
heatmap_matrix = heatmap_matrix[:, sorted_col_idx]
row_labels = [row_labels[i] for i in sorted_row_idx]
col_labels = [col_labels[i] for i in sorted_col_idx]

# Assign colors to time periods
time_periods = pivot_df.index.get_level_values(0).unique()
time_colors = {time: f"rgb({np.random.randint(50, 255)}, {np.random.randint(50, 255)}, {np.random.randint(50, 255)})" for time in time_periods}
row_time_periods = [pivot_df.index[i][0] for i in sorted_row_idx]  # Get reordered time periods
row_colors = [time_colors[time] for time in row_time_periods]

# Create the heatmap using Plotly
fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_matrix,
        x=col_labels,
        y=row_labels,
        colorscale="Viridis",
        colorbar=dict(title=f"SHAP Value"),  # Properly define colorbar title
        text=np.round(heatmap_matrix, 2),  # Rounded numbers for annotations
        hoverinfo="text+z"  # Display text and SHAP values on hover
    )
)

# Add colored rectangles for time periods
for i, color in enumerate(row_colors):
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=-0.4,  # Small vertical bar to the left of the heatmap
        y0=i - 0.5, y1=i + 0.5,
        line=dict(width=0),
        fillcolor=color,
    )

# Add a legend for time period colors below the heatmap
time_period_legend = [
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=10, color=color),
        name=time,
        showlegend=True
    )
    for time, color in time_colors.items()
]

# Add the time period legend as invisible scatter traces
fig.add_traces(time_period_legend)

# Update layout for better spacing of legends
fig.update_layout(
    title="Hierarchical SHAP Heatmap",
    title_x=0.5,
    xaxis=dict(
        title="Cluster",
        side="bottom",  # Move x-axis labels to the bottom
        title_font=dict(size=16),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title="Time Period, Frequency",
        title_font=dict(size=16),
        tickfont=dict(size=12)
    ),
    margin=dict(l=150, r=50, t=50, b=150),  # Adjust margins for better spacing
    height=800,
    width=800,
    legend=dict(
        orientation="h",  # Horizontal legend for time periods
        y=-0.2,  # Position below the heatmap
        x=0.5,
        xanchor="center",
        title='Time Period'
    )
)

# Display the heatmap in Streamlit
st.plotly_chart(fig, use_container_width=False)