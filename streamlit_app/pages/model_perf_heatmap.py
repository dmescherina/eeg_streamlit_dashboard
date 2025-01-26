# pages/model_perf_heatmap.py
# Title: Model Cross-Performance Heatmap

import streamlit as st
import pandas as pd
import plotly.express as px
import config

# Set Streamlit page layout to wide
st.set_page_config(page_title="Model Cross-Performance", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_FILE_MODEL_PERF, skiprows=1)

# Load the data
df = load_data()

# Set index to match the original
df.index = df.columns

# Streamlit App Title
st.title("Interactive Model Performance Heatmap")

# Create an interactive heatmap using Plotly
fig = px.imshow(
    df.values,  # Heatmap data
    labels=dict(x="Columns", y="Rows", color="Value"),  # Label names
    x=df.columns,  # Column names
    y=df.index,  # Row names
    color_continuous_scale="viridis",  # Color scale
    zmin=0.5,
    text_auto=".2f",  # Display values with 2 decimal places
)

# Update layout for better readability
fig.update_layout(
    width=1000,  # Set width
    height=1000,  # Set height
    margin=dict(l=50, r=50, t=50, b=50),  # Margins around the heatmap
    title="Model Performance Heatmap",
    title_x=0.5,  # Center the title
    font=dict(size=14),  # Increase font size for overall labels
    xaxis=dict(title="Columns", title_font=dict(size=18), tickfont=dict(size=14)),  # X-axis font sizes
    yaxis=dict(title="Rows", title_font=dict(size=18), tickfont=dict(size=14)),  # Y-axis font sizes
)

# Display the interactive heatmap in Streamlit
st.plotly_chart(fig, use_container_width=False)  # Disable container width to respect explicit size