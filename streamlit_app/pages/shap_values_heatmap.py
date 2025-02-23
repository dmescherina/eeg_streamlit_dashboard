import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import config

# Set Streamlit page layout to wide
st.set_page_config(page_title="Fixed Order SHAP Heatmap", layout="wide")

# Define fixed orders and labels
FREQUENCY_ORDER = ['delta', 'theta', 'alpha', 'beta', 'lower gamma']
FREQUENCY_RANGES = {
    'delta': '1-4 Hz',
    'theta': '4-8 Hz',
    'alpha': '8-12 Hz',
    'beta': '12-30 Hz',
    'lower gamma': '30-48 Hz'
}

TIME_PERIOD_ORDER = ['(-4, 0)', '(0, 5)', '(5, 10)']
TIME_PERIOD_LABELS = {
    '(-4, 0)': 'Pre-stimulus',
    '(0, 5)': 'Early',
    '(5, 10)': 'Late'
}

CLUSTER_ORDER = ['CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']
CLUSTER_LABELS = {
    'CL1': 'Left Frontal',
    'CL4': 'Right Frontal',
    'CL2': 'Left Central',
    'CL5': 'Right Central',
    'CL3': 'Left Posterior',
    'CL6': 'Right Posterior'
}

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# UI Elements
st.title("SHAP Values Heatmap")

data_source = st.radio(
    "Select Data Source",
    options=["Averaged SHAP Values", "Summed SHAP Values"],
    index=0
)

if data_source == "Averaged SHAP Values":
    file_path = config.DATA_FILE_SHAP_MEAN
    aggregation_method = "mean"
else:
    file_path = config.DATA_FILE_SHAP_SUM
    aggregation_method = "sum"

df = load_data(file_path)

# Sidebar Filters
st.sidebar.header("Filters")
selected_clusters = st.sidebar.multiselect(
    "Select Clusters",
    options=CLUSTER_ORDER,
    default=CLUSTER_ORDER
)
selected_time_periods = st.sidebar.multiselect(
    "Select Time Periods",
    options=TIME_PERIOD_ORDER,
    default=TIME_PERIOD_ORDER
)
selected_frequencies = st.sidebar.multiselect(
    "Select Frequencies",
    options=FREQUENCY_ORDER,
    default=FREQUENCY_ORDER
)
selected_stimuli = st.sidebar.multiselect(
    "Select Stimulus",
    options=df['Stimulus'].unique(),
    default=df['Stimulus'].unique()
)
selected_scores = st.sidebar.multiselect(
    "Select Target Scores",
    options=df['TargetScore'].unique(),
    default=df['TargetScore'].unique()
)

# Filter the DataFrame
filtered_df = df[
    (df['Cluster'].isin(selected_clusters)) &
    (df['Time Period'].isin(selected_time_periods)) &
    (df['Frequency'].isin(selected_frequencies)) &
    (df['Stimulus'].isin(selected_stimuli)) &
    (df['TargetScore'].isin(selected_scores))
]

# Create categories with fixed ordering
filtered_df['Time Period'] = pd.Categorical(
    filtered_df['Time Period'],
    categories=TIME_PERIOD_ORDER,
    ordered=True
)
filtered_df['Frequency'] = pd.Categorical(
    filtered_df['Frequency'],
    categories=FREQUENCY_ORDER,
    ordered=True
)
filtered_df['Cluster'] = pd.Categorical(
    filtered_df['Cluster'],
    categories=CLUSTER_ORDER,
    ordered=True
)

# Ensure proper data types and create pivot table
filtered_df['SHAP_value'] = pd.to_numeric(filtered_df['SHAP_value'], errors='coerce')
pivot_df = filtered_df.pivot_table(
    index=["Time Period", "Frequency"],
    columns="Cluster",
    values="SHAP_value",
    aggfunc=aggregation_method,
    fill_value=0
)

# Reorder with our fixed ordering
pivot_df = pivot_df.reindex(
    index=pd.MultiIndex.from_product([TIME_PERIOD_ORDER, FREQUENCY_ORDER]),
    columns=CLUSTER_ORDER,
    fill_value=0
)

# Prepare data for heatmap
heatmap_matrix = pivot_df.values
row_labels = [f"{TIME_PERIOD_LABELS[time]}, {freq} ({FREQUENCY_RANGES[freq]})" for time, freq in pivot_df.index]
col_labels = pivot_df.columns.tolist()

# Create color scheme for time periods
time_colors = {
    '(-4, 0)': 'rgb(70, 130, 180)',   # Steel Blue
    '(0, 5)': 'rgb(60, 179, 113)',    # Medium Sea Green
    '(5, 10)': 'rgb(238, 130, 238)'   # Violet
}

# Get time period for each row
row_time_periods = [pivot_df.index[i][0] for i in range(len(pivot_df.index))]
row_colors = [time_colors[time] for time in row_time_periods]

# Create the heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_matrix,
        x=col_labels,
        y=row_labels,
        colorscale="Viridis",
        colorbar=dict(
            title=f"SHAP Value ({aggregation_method})",
            titleside="right",
            title_font=dict(color='white')
        ),
        text=np.round(heatmap_matrix, 3),
        hovertemplate="Cluster: %{x}<br>Time Period & Frequency: %{y}<br>SHAP Value: %{z:.3f}<extra></extra>"
    )
)

# Add colored rectangles for time periods
for i, color in enumerate(row_colors):
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=-0.4,
        y0=i - 0.5,
        y1=i + 0.5,
        line=dict(width=0),
        fillcolor=color,
    )

# Add legend for time periods
time_period_legend = [
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=10, color=color),
        name=TIME_PERIOD_LABELS[time],
        showlegend=True
    )
    for time, color in time_colors.items()
]
fig.add_traces(time_period_legend)

# Update layout
fig.update_layout(
    title=dict(
        text="SHAP Values Distribution Across Brain Regions",
        x=0.5,
        font=dict(size=20, color='black')
    ),
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis=dict(
        title="Clusters (Left → Right, Front → Back)",
        title_font=dict(size=16, color='black'),
        tickfont=dict(size=12, color='black'),
        gridcolor='#444444',
        showgrid=True,
        ticktext=[f"{CLUSTER_LABELS[cl]} ({cl})" for cl in CLUSTER_ORDER],
        tickvals=col_labels
    ),
    yaxis=dict(
        title="Time Period, Frequency",
        title_font=dict(size=16, color='black'),
        tickfont=dict(size=12, color='black'),
        gridcolor='#444444',
        showgrid=True
    ),
    margin=dict(l=150, r=50, t=100, b=150),
    height=800,
    width=1000,
    legend=dict(
        title="Time Period",
        title_font=dict(color='black'),
        font=dict(color='black'),
        orientation="h",
        y=-0.2,
        x=0.5,
        xanchor="center",
        bgcolor='white'
    )
)

# Display the heatmap
st.plotly_chart(fig, use_container_width=True)

# Add explanatory text
st.markdown("""
### Cluster Information
- **Frontal Clusters**: CL1 (Left) and CL4 (Right)
- **Central Clusters**: CL2 (Left) and CL5 (Right)
- **Posterior Clusters**: CL3 (Left) and CL6 (Right)

### Frequency Bands
- Delta: 1-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-12 Hz
- Beta: 12-30 Hz
- Lower Gamma: 30-48 Hz

### Time Periods
- Pre-stimulus: -4 to 0 seconds
- Early: 0 to 5 seconds
- Late: 5 to 10 seconds
""")