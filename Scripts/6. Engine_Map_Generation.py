'''
0. README
Notebook: Engine_Map_TCO&HP_Generation
Inputs
•	customer_data_analytics.pt_integration.engine_fuel_map_interpolated
Outputs
•	pt_configuration_thesis.alltrucks_v1.TCO_Fuel_Map_50_50
•	pt_configuration_thesis.alltrucks_v1.HP_Fuel_Map_50_50
Purpose
•	This notebook is meant to take as input the existing interpolated engine fuel map data which is non uniform and generate a grid style - uniform SFC map (50 RPM X 50 Nm) with each grid having a designated SFC value. This is a preparatory step for LEMP calculation.
•	The above process is repeated for both TCO and HP maps.
Other remarks
•	Other interpolation methods than the one used in this notebook can be tried if required.
'''


# 1.	HEADERS
# PySpark Headers
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.functions import col, desc
# Plotly Headers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Pandas Headers 
import pandas as pd 
# matplotlib headers 
import matplotlib.pyplot as plt

# 2.	FUEL MAP INPUT
df_fcm = spark.table("customer_data_analytics.pt_integration.engine_fuel_map_interpolated")
display(df_fcm)
from pyspark.sql import functions as F

# Number of unique entries
unique_count = df_fcm.select("engine_map_name").distinct().count()

# Unique entries
unique_entries = df_fcm.select("engine_map_name").distinct()

print("Number of unique engine_map_name values:", unique_count)
unique_entries.show(truncate=False)

# 3.	EXTRACT EITHER TCO OR HP MAP
fcm_interpolated = df_fcm.filter(f.col("engine_map_name") == "OM471HDEP2020TCO_TMH").toPandas()

display(fcm_interpolated)

fcm_interpolated.info()

# 4.	PLOT CONTOUR
# To visualize the existing non-unform map

import plotly.graph_objects as go

# Create a new figure
fig_heatmap = go.Figure()

# Add the contour trace
fig_heatmap.add_trace(go.Contour(
    x=fcm_interpolated['engine_speed'],
    y=fcm_interpolated['torque'],
    z=fcm_interpolated['spec_fuel_cons'],
    line=dict(smoothing=1.3),  # smooth contour lines
    contours=dict(
        coloring='lines',
        showlabels=True,
        labelformat=".0f",
        start=fcm_interpolated['spec_fuel_cons'].min(),
        end=230,
        size=0.5
    ),
    name='Specific Fuel Consumption',
    showscale=False,
    hovertemplate='Engine Speed: %{x}<br>Torque: %{y}<br>SpecFuelConsumption: %{z}<extra></extra>'
))


fig_heatmap.update_layout(
    title='Engine Specific Fuel Consumption Contours',
    xaxis_title='Engine Speed (RPM)',
    yaxis_title='Torque (Nm)'
)

# Show the figure
fig_heatmap.show()

# 5.	DEFINE BIN GRID AND COMPUTE/ INTERPOLATE MEAN SFC PER BIN GRID
max_rpm = fcm_interpolated["engine_speed"].max()
max_torque = fcm_interpolated["torque"].max()

import numpy as np
import pandas as pd

# Step 1: Define bins (50 RPM X 50 Nm)
rpm_bins = np.arange(0, max_rpm + 50, 50)       # from 0 to max_rpm, step 50
torque_bins = np.arange(0, max_torque + 50, 50)  # from 0 to max_torque, step 100

# Step 2: Assign each value to a bin
fcm_interpolated["rpm_bin"] = pd.cut(fcm_interpolated["engine_speed"], bins=rpm_bins, right=False)
fcm_interpolated["torque_bin"] = pd.cut(fcm_interpolated["torque"], bins=torque_bins, right=False)

# Step 3: Compute mean fuel consumption per bin
box_avg = fcm_interpolated.groupby(["rpm_bin", "torque_bin"], as_index=False)["spec_fuel_cons"].mean()

# Step 4: Pivot to 2D grid (RPM rows, Torque columns)
fuel_map = box_avg.pivot(index="rpm_bin", columns="torque_bin", values="spec_fuel_cons")

# Optional: display
print(fuel_map)


import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

plt.figure(figsize=(14, 8))
sns.heatmap(
    fuel_map.T,           # transpose so RPM is on X and Torque on Y
    cmap="viridis",
    annot=False,
    cbar_kws={'label': 'Average Fuel Consumption'},
    linewidths=0.5
)

# Labels
plt.title("Fuel Consumption Map (SFC) [g/kWh]")
plt.xlabel("RPM Bin [rpm]")
plt.ylabel("Torque Bin [Nm]")

# Rotate x-axis labels for clarity
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Flip Y-axis so Torque increases upward
plt.gca().invert_yaxis()

plt.show()
# Convert interval objects to strings for readability
fuel_map_readable = fuel_map.copy()
fuel_map_readable.index = fuel_map_readable.index.astype(str)
fuel_map_readable.columns = fuel_map_readable.columns.astype(str)

# Display
fuel_map_readable

# 6.	INTERPOLATE FOR MISSING BINS
fuel_map_interpolated = fuel_map.interpolate(axis=0, method='linear') \
                                  .interpolate(axis=1, method='linear')

fuel_map_interpolated.columns = [
    f"{int(col.left)}_{int(col.right)}" if isinstance(col, pd.Interval) else col
    for col in fuel_map_interpolated.columns
]

print(fuel_map_interpolated)

display(fuel_map_interpolated)

# Convert interval objects to strings for readability
fuel_map_interpolated_readable = fuel_map_interpolated.copy()
fuel_map_interpolated_readable.index = fuel_map_interpolated_readable.index.astype(str)
fuel_map_interpolated_readable.columns = fuel_map_interpolated_readable.columns.astype(str)

# Display
fuel_map_interpolated_readable

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

plt.figure(figsize=(14, 8))
sns.heatmap(
    fuel_map_interpolated.T,           # transpose so RPM is on X and Torque on Y
    cmap="viridis",
    annot=False,
    cbar_kws={'label': 'Average Fuel Consumption'},
    linewidths=0.5
)

# Labels
plt.title("Fuel Consumption Map (SFC) [g/kWh]")
plt.xlabel("RPM Bin [rpm]")
plt.ylabel("Torque Bin [Nm]")

# Rotate x-axis labels for clarity
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Flip Y-axis so Torque increases upward
plt.gca().invert_yaxis()

plt.show()


# 7.	PUSH ESSENTIAL DATA TO CATALOG
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
fuel_map_interpolated_spark = spark.createDataFrame(fuel_map_interpolated)
fuel_map_interpolated_spark.write.mode("overwrite").saveAsTable("pt_configuration_thesis.alltrucks_v1.HP_Fuel_Map_50_50")
