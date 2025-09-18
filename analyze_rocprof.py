import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Read the stats CSV file
stats_df = pd.read_csv('results.stats.csv')

# Drop unwanted kernel entries
drop_names = [
    # "<barrier packet>",
    # "pack_convert_fp32_to_bf16_tiles16_kernel(float const*, int, int, int, int, hip_bfloat16*, int, int)",
]
stats_df = stats_df[~stats_df['Name'].isin(drop_names)].copy()

# Sort by total time
stats_df = stats_df.sort_values(by='TotalDurationNs', ascending=False)

# Convert nanoseconds to milliseconds
stats_df['DurationMs'] = stats_df['TotalDurationNs'] / 1_000_000
stats_df['AverageMs'] = stats_df['AverageNs'] / 1_000_000

# Create bar chart for kernel execution times
fig = px.bar(
    stats_df.head(15),
    x='Name',
    y='DurationMs',
    title='Top 15 Most Time-Consuming GPU Kernels',
    labels={'DurationMs': 'Duration (ms)', 'KernelName': 'Kernel Name'},
    height=600,
    hover_data=['Calls']  # show number of calls in hover
)

# Rotate x-axis labels for better readability
fig.update_layout(xaxis_tickangle=-45)

# Save the plot as HTML
# fig.write_html('kernel_profile.html')

# Create pie chart for percentage of total time
total_time = stats_df['DurationMs'].sum()
print(total_time)
stats_df['Percentage'] = (stats_df['DurationMs'] / total_time) * 100
top_df = stats_df.head(10).copy()
# Pre-format hover details for duration and calls
top_df["HoverText"] = top_df.apply(
    lambda r: f"Duration: {r['DurationMs']:.2f} ms<br>Calls: {int(r['Calls'])}<br>AverageMs: {int(r['AverageMs'])}", axis=1
)

fig_pie = px.pie(
    top_df,
    values='Percentage',
    names='Name',
    title='Distribution of Execution Time (Top 10 Kernels)',
    # Keep default hover data minimal; we supply our own text
    hover_data=None
)

# Show only percentage on-slice, but add DurationMs and Calls in hover
fig_pie.update_traces(
    text=top_df["HoverText"],
    textinfo="percent",
    hovertemplate=(
        "Name: %{label}<br>"
        "Percentage: %{value:.2f}%<br>"
        "%{text}<extra></extra>"
    )
)
fig_pie.write_html(f'kernel_distribution.html')
