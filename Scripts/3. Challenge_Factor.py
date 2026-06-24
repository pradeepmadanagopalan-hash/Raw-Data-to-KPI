'''
0. README
Notebook: Challenge_Factor
Inputs
•	pt_configuration_thesis.alltrucks_v1.weight_mds_results
•	pt_configuration_thesis.alltrucks_v1.gradient_mds_results
•	pt_configuration_thesis.alltrucks_v1.vin_info_sg_info_cluster_info_talpy_info
•	pt_configuration_thesis.alltrucks_v1.combined_weight_spectra_outlier_removed
•	pt_configuration_thesis.alltrucks_v1.combined_gradient_spectra_outlier_removed
Outputs
•	pt_configuration_thesis.alltrucks_v1.vin_info_sg_info_cluster_info_talpy_info_MDS_info_CF_info
Purpose
•	This notebook is meant to take as input the downscaled MDS matrices of weight and gradient from the notebooks MDS_Weight and MDS_Gradient, pass these to a linear regression model and generate a KPI called Challenge Factor (CF).
•	It then creates a global dataframe combining all the results generated in previous notebook with the MDS embeddings and CF to create vin_info_sg_info_cluster_info_talpy_info_MDS_info_CF_info.
Other remarks
•	Directly skip to section - 9. It has the linear regression model. Sections before that are meant to interpret the results from MDS notebooks to improve understanding of the technique.
'''


# 1.	HEADERS
import pyspark.sql.functions as f
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from functools import reduce

import datetime
import warnings

from talpy.timeseries import ts_transformer, ts_column_factory
from talpy.helper_functions import table_helper

# 2.	INPUT DATA
weight_MDS = spark.table("pt_configuration_thesis.alltrucks_v1.`weight_mds_results`")

unique_vins_count_ori = weight_MDS.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

gradient_MDS = spark.table("pt_configuration_thesis.alltrucks_v1.`gradient_mds_results`")

unique_vins_count_ori = gradient_MDS.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

display(weight_MDS)
display(gradient_MDS)


info = spark.table("pt_configuration_thesis.alltrucks_v1.`vin_info_sg_info_cluster_info_talpy_info`")

unique_vins_count_ori = info.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

weight_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`combined_weight_spectra_outlier_removed`")

unique_vins_count_ori = weight_spectra.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

gradient_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`combined_gradient_spectra_outlier_removed`")

unique_vins_count_ori = gradient_spectra.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

# 3.	INTERPRETABILITY - WEIGHT - MDS1_w
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Compute quantiles
quantiles = weight_MDS.approxQuantile("MDS1_w", [0.2, 0.4, 0.6, 0.8], 0.001)
q20, q40, q60, q80 = quantiles

# Add class label from 1 to 5
w_mds_classed = weight_MDS.withColumn(
    "MDS1_class",
    F.when(F.col("MDS1_w") <= q20, 1)
     .when(F.col("MDS1_w") <= q40, 2)
     .when(F.col("MDS1_w") <= q60, 3)
     .when(F.col("MDS1_w") <= q80, 4)
     .otherwise(5)
)

from pyspark.sql import Window

# Prepare spectra (lower interval bound + normalization)
ws = weight_spectra.withColumn(
    "x_bin",
    F.regexp_extract("x1_interval", r"\[(\d+),", 1).cast("int")
)

ws_norm = ws.withColumn(
    "total_count",
    F.sum("count").over(Window.partitionBy("vin"))
).withColumn(
    "norm_count",
    F.col("count") / F.col("total_count")
)

ws_joined = ws_norm.join(
    w_mds_classed.select("vin", "MDS1_class"),
    on="vin",
    how="inner"
)

avg_per_class = (
    ws_joined.groupBy("MDS1_class", "x_bin")
             .agg(F.avg("norm_count").alias("avg_norm"))
)

pdf = avg_per_class.toPandas()
pdf = pdf.sort_values(["MDS1_class", "x_bin"])

class_counts = (
    w_mds_classed.groupBy("MDS1_class")
                 .agg(F.countDistinct("vin").alias("n_trucks"))
                 .orderBy("MDS1_class")
)

class_counts.show()

pdf_counts = class_counts.toPandas().set_index("MDS1_class")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for cls in range(1, 6):
    grp = pdf[pdf["MDS1_class"] == cls]
    n_cls = pdf_counts.loc[cls, "n_trucks"]  # lookup count

    plt.plot(
        grp["x_bin"],
        grp["avg_norm"],
        linewidth=2,
        label=f"Class {cls} ({n_cls} trucks)"
    )

plt.xlabel("Weight bin lower bound (kg)")
plt.ylabel("Average normalized count")
plt.title("Class-wise Average Weight Spectra — binned by MDS1_w")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4.	INTERPRETABILITY - WEIGHT - MDS2_w
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# Step 1: Compute quantiles for MDS2_w
# -----------------------------
quantiles = weight_MDS.approxQuantile("MDS2_w", [0.2, 0.4, 0.6, 0.8], 0.001)
q20, q40, q60, q80 = quantiles

# -----------------------------
# Step 2: Assign class labels 1-5 based on MDS2_w
# -----------------------------
w_mds_classed = weight_MDS.withColumn(
    "MDS2_class",
    F.when(F.col("MDS2_w") <= q20, 1)
     .when(F.col("MDS2_w") <= q40, 2)
     .when(F.col("MDS2_w") <= q60, 3)
     .when(F.col("MDS2_w") <= q80, 4)
     .otherwise(5)
)

# -----------------------------
# Step 3: Prepare weight spectra (parse bins + normalize)
# -----------------------------
ws = weight_spectra.withColumn(
    "x_bin",
    F.regexp_extract("x1_interval", r"\[(\d+),", 1).cast("int")
)

ws_norm = ws.withColumn(
    "total_count",
    F.sum("count").over(Window.partitionBy("vin"))
).withColumn(
    "norm_count",
    F.col("count") / F.col("total_count")
)

# -----------------------------
# Step 4: Join spectra with MDS2 class
# -----------------------------
ws_joined = ws_norm.join(
    w_mds_classed.select("vin", "MDS2_class"),
    on="vin",
    how="inner"
)

# -----------------------------
# Step 5: Compute average spectrum per class
# -----------------------------
avg_per_class = (
    ws_joined.groupBy("MDS2_class", "x_bin")
             .agg(F.avg("norm_count").alias("avg_norm"))
)

pdf = avg_per_class.toPandas()
pdf = pdf.sort_values(["MDS2_class", "x_bin"])

# -----------------------------
# Step 6: Compute number of trucks per class
# -----------------------------
class_counts = (
    w_mds_classed.groupBy("MDS2_class")
                 .agg(F.countDistinct("vin").alias("n_trucks"))
                 .orderBy("MDS2_class")
)

class_counts.show()

pdf_counts = class_counts.toPandas().set_index("MDS2_class")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for cls in range(1, 6):
    grp = pdf[pdf["MDS2_class"] == cls]      # changed to MDS2_class
    n_cls = pdf_counts.loc[cls, "n_trucks"]  # lookup count remains the same

    plt.plot(
        grp["x_bin"],
        grp["avg_norm"],
        linewidth=2,
        label=f"Class {cls} ({n_cls} trucks)"
    )

plt.xlabel("Weight bin lower bound (kg)")
plt.ylabel("Average normalized count")
plt.title("Class-wise Average Weight Spectra — binned by MDS2_w")  # updated title
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5.	INTERPRETABILITY - WEIGHT - MDS3_w
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# Step 1: Compute quantiles for MDS3_w
# -----------------------------
quantiles = weight_MDS.approxQuantile("MDS3_w", [0.2, 0.4, 0.6, 0.8], 0.001)
q20, q40, q60, q80 = quantiles

# -----------------------------
# Step 2: Assign class labels 1-5 based on MDS3_w
# -----------------------------
w_mds_classed = weight_MDS.withColumn(
    "MDS3_class",
    F.when(F.col("MDS3_w") <= q20, 1)
     .when(F.col("MDS3_w") <= q40, 2)
     .when(F.col("MDS3_w") <= q60, 3)
     .when(F.col("MDS3_w") <= q80, 4)
     .otherwise(5)
)

# -----------------------------
# Step 3: Prepare weight spectra (parse bins + normalize)
# -----------------------------
ws = weight_spectra.withColumn(
    "x_bin",
    F.regexp_extract("x1_interval", r"\[(\d+),", 1).cast("int")
)

ws_norm = ws.withColumn(
    "total_count",
    F.sum("count").over(Window.partitionBy("vin"))
).withColumn(
    "norm_count",
    F.col("count") / F.col("total_count")
)

# -----------------------------
# Step 4: Join spectra with MDS3 class
# -----------------------------
ws_joined = ws_norm.join(
    w_mds_classed.select("vin", "MDS3_class"),
    on="vin",
    how="inner"
)

# -----------------------------
# Step 5: Compute average spectrum per class
# -----------------------------
avg_per_class = (
    ws_joined.groupBy("MDS3_class", "x_bin")
             .agg(F.avg("norm_count").alias("avg_norm"))
)

pdf = avg_per_class.toPandas()
pdf = pdf.sort_values(["MDS3_class", "x_bin"])

# -----------------------------
# Step 6: Compute number of trucks per class
# -----------------------------
class_counts = (
    w_mds_classed.groupBy("MDS3_class")
                 .agg(F.countDistinct("vin").alias("n_trucks"))
                 .orderBy("MDS3_class")
)

class_counts.show()

pdf_counts = class_counts.toPandas().set_index("MDS3_class")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for cls in range(1, 6):
    grp = pdf[pdf["MDS3_class"] == cls]
    n_cls = pdf_counts.loc[cls, "n_trucks"]

    plt.plot(
        grp["x_bin"],
        grp["avg_norm"],
        linewidth=2,
        label=f"Class {cls} ({n_cls} trucks)"
    )

plt.xlabel("Weight bin lower bound (kg)")
plt.ylabel("Average normalized count")
plt.title("Class-wise Average Weight Spectra — binned by MDS3_w")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6.	INTERPRETABILITY - GRADIENT - MDS1_g
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# Step 1: Compute quantiles for MDS1_g
# -----------------------------
quantiles = gradient_MDS.approxQuantile("MDS1_g", [0.2, 0.4, 0.6, 0.8], 0.001)
q20, q40, q60, q80 = quantiles

# -----------------------------
# Step 2: Assign class labels 1-5 based on MDS1_g
# -----------------------------
g_mds_classed = gradient_MDS.withColumn(
    "MDS1_g_class",
    F.when(F.col("MDS1_g") <= q20, 1)
     .when(F.col("MDS1_g") <= q40, 2)
     .when(F.col("MDS1_g") <= q60, 3)
     .when(F.col("MDS1_g") <= q80, 4)
     .otherwise(5)
)

# -----------------------------
# Step 3: Prepare gradient spectra (parse bins + normalize)
# -----------------------------
gs = gradient_spectra.withColumn(
    "x_bin",
    F.regexp_extract("x1_interval", r"\[(\d+),", 1).cast("int")
)

gs_norm = gs.withColumn(
    "total_count",
    F.sum("count").over(Window.partitionBy("vin"))
).withColumn(
    "norm_count",
    F.col("count") / F.col("total_count")
)

# -----------------------------
# Step 4: Join spectra with MDS1_g class
# -----------------------------
gs_joined = gs_norm.join(
    g_mds_classed.select("vin", "MDS1_g_class"),
    on="vin",
    how="inner"
)

# -----------------------------
# Step 5: Compute average spectrum per class
# -----------------------------
avg_per_class = (
    gs_joined.groupBy("MDS1_g_class", "x_bin")
             .agg(F.avg("norm_count").alias("avg_norm"))
)

pdf = avg_per_class.toPandas()
pdf = pdf.sort_values(["MDS1_g_class", "x_bin"])

# -----------------------------
# Step 6: Compute number of trucks per class
# -----------------------------
class_counts = (
    g_mds_classed.groupBy("MDS1_g_class")
                 .agg(F.countDistinct("vin").alias("n_trucks"))
                 .orderBy("MDS1_g_class")
)

class_counts.show()

pdf_counts = class_counts.toPandas().set_index("MDS1_g_class")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for cls in range(1, 6):
    grp = pdf[pdf["MDS1_g_class"] == cls]
    n_cls = pdf_counts.loc[cls, "n_trucks"]

    plt.plot(
        grp["x_bin"],
        grp["avg_norm"],
        linewidth=2,
        label=f"Class {cls} ({n_cls} trucks)"
    )

plt.xlabel("Gradient bin lower bound")
plt.ylabel("Average normalized count")
plt.title("Class-wise Average Gradient Spectra — binned by MDS1_g")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7.	INTERPRETABILITY - GRADIENT - MDS2_g
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# Step 1: Compute quantiles for MDS2_g
# -----------------------------
quantiles = gradient_MDS.approxQuantile("MDS2_g", [0.2, 0.4, 0.6, 0.8], 0.001)
q20, q40, q60, q80 = quantiles

# -----------------------------
# Step 2: Assign class labels 1-5 based on MDS2_g
# -----------------------------
g_mds_classed = gradient_MDS.withColumn(
    "MDS2_g_class",
    F.when(F.col("MDS2_g") <= q20, 1)
     .when(F.col("MDS2_g") <= q40, 2)
     .when(F.col("MDS2_g") <= q60, 3)
     .when(F.col("MDS2_g") <= q80, 4)
     .otherwise(5)
)

# -----------------------------
# Step 3: Prepare gradient spectra (parse bins + normalize)
# -----------------------------
gs = gradient_spectra.withColumn(
    "x_bin",
    F.regexp_extract("x1_interval", r"\[(\d+),", 1).cast("int")
)

gs_norm = gs.withColumn(
    "total_count",
    F.sum("count").over(Window.partitionBy("vin"))
).withColumn(
    "norm_count",
    F.col("count") / F.col("total_count")
)

# -----------------------------
# Step 4: Join spectra with MDS2_g class
# -----------------------------
gs_joined = gs_norm.join(
    g_mds_classed.select("vin", "MDS2_g_class"),
    on="vin",
    how="inner"
)

# -----------------------------
# Step 5: Compute average spectrum per class
# -----------------------------
avg_per_class = (
    gs_joined.groupBy("MDS2_g_class", "x_bin")
             .agg(F.avg("norm_count").alias("avg_norm"))
)

pdf = avg_per_class.toPandas()
pdf = pdf.sort_values(["MDS2_g_class", "x_bin"])

# -----------------------------
# Step 6: Compute number of trucks per class
# -----------------------------
class_counts = (
    g_mds_classed.groupBy("MDS2_g_class")
                 .agg(F.countDistinct("vin").alias("n_trucks"))
                 .orderBy("MDS2_g_class")
)

class_counts.show()

pdf_counts = class_counts.toPandas().set_index("MDS2_g_class")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for cls in range(1, 6):
    grp = pdf[pdf["MDS2_g_class"] == cls]
    n_cls = pdf_counts.loc[cls, "n_trucks"]

    plt.plot(
        grp["x_bin"],
        grp["avg_norm"],
        linewidth=2,
        label=f"Class {cls} ({n_cls} trucks)"
    )

plt.xlabel("Gradient bin lower bound")
plt.ylabel("Average normalized count")
plt.title("Class-wise Average Gradient Spectra — binned by MDS2_g")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 8.	INTERPRETABILITY - GRADIENT - MDS3_g
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# Step 1: Compute quantiles for MDS3_g
# -----------------------------
quantiles = gradient_MDS.approxQuantile("MDS3_g", [0.2, 0.4, 0.6, 0.8], 0.001)
q20, q40, q60, q80 = quantiles

# -----------------------------
# Step 2: Assign class labels 1-5 based on MDS3_g
# -----------------------------
g_mds_classed = gradient_MDS.withColumn(
    "MDS3_g_class",
    F.when(F.col("MDS3_g") <= q20, 1)
     .when(F.col("MDS3_g") <= q40, 2)
     .when(F.col("MDS3_g") <= q60, 3)
     .when(F.col("MDS3_g") <= q80, 4)
     .otherwise(5)
)

# -----------------------------
# Step 3: Prepare gradient spectra (parse bins + normalize)
# -----------------------------
gs = gradient_spectra.withColumn(
    "x_bin",
    F.regexp_extract("x1_interval", r"\[(\d+),", 1).cast("int")
)

gs_norm = gs.withColumn(
    "total_count",
    F.sum("count").over(Window.partitionBy("vin"))
).withColumn(
    "norm_count",
    F.col("count") / F.col("total_count")
)

# -----------------------------
# Step 4: Join spectra with MDS3_g class
# -----------------------------
gs_joined = gs_norm.join(
    g_mds_classed.select("vin", "MDS3_g_class"),
    on="vin",
    how="inner"
)

# -----------------------------
# Step 5: Compute average spectrum per class
# -----------------------------
avg_per_class = (
    gs_joined.groupBy("MDS3_g_class", "x_bin")
             .agg(F.avg("norm_count").alias("avg_norm"))
)

pdf = avg_per_class.toPandas()
pdf = pdf.sort_values(["MDS3_g_class", "x_bin"])

# -----------------------------
# Step 6: Compute number of trucks per class
# -----------------------------
class_counts = (
    g_mds_classed.groupBy("MDS3_g_class")
                 .agg(F.countDistinct("vin").alias("n_trucks"))
                 .orderBy("MDS3_g_class")
)

class_counts.show()

pdf_counts = class_counts.toPandas().set_index("MDS3_g_class")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for cls in range(1, 6):
    grp = pdf[pdf["MDS3_g_class"] == cls]
    n_cls = pdf_counts.loc[cls, "n_trucks"]

    plt.plot(
        grp["x_bin"],
        grp["avg_norm"],
        linewidth=2,
        label=f"Class {cls} ({n_cls} trucks)"
    )

plt.xlabel("Gradient bin lower bound")
plt.ylabel("Average normalized count")
plt.title("Class-wise Average Gradient Spectra — binned by MDS3_g")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 9.	BUILD CHALLENGE FACTOR - LINEAR REGRESSION
# Select MDS embeddings and fuel consumption
df_cf = weight_MDS.select(
    "vin", "MDS1_w", "MDS2_w", "MDS3_w"
).join(
    gradient_MDS.select("vin", "MDS1_g", "MDS2_g", "MDS3_g"),
    on="vin",
    how="inner"
).join(
    info.select("vin", "fuel_cons_l_per_100km"),
    on="vin",
    how="inner"
)

df_cf_pd = df_cf.toPandas()

from sklearn.preprocessing import StandardScaler

X = df_cf_pd[["MDS1_w", "MDS2_w", "MDS3_w",
              "MDS1_g", "MDS2_g", "MDS3_g"]].copy()
y = df_cf_pd["fuel_cons_l_per_100km"].values

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled.iloc[:, :6] = scaler.fit_transform(X.iloc[:, :6])

from sklearn.linear_model import LinearRegression
import numpy as np

lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

coefficients = lr_model.coef_
intercept = lr_model.intercept_

# Optional: check coefficients
for feat, coef in zip(X_scaled.columns, coefficients):
    print(f"{feat}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")

df_cf_pd['CF_raw'] = np.dot(X_scaled, coefficients)

df_cf_pd['CF'] = (df_cf_pd['CF_raw'] - df_cf_pd['CF_raw'].min()) / \
                 (df_cf_pd['CF_raw'].max() - df_cf_pd['CF_raw'].min())

n_bins = 5
df_cf_pd['CF_bin'] = pd.cut(df_cf_pd['CF'], bins=n_bins, labels=False)

print(df_cf_pd['CF_bin'].value_counts())

from sklearn.metrics import r2_score

# Predicted values using the linear model
y_pred = np.dot(X_scaled, coefficients) + intercept

# Compute R²
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.4f}")

# 10.	JOIN ALL ESSENTIAL INFO FROM THIS NOTEBOOK
# Add embeddings and fuel consumption to the info table

info_extended = (
    info
    .join(
        weight_MDS.select("vin", "MDS1_w", "MDS2_w", "MDS3_w"),
        on="vin",
        how="left"
    )
    .join(
        gradient_MDS.select("vin", "MDS1_g", "MDS2_g", "MDS3_g"),
        on="vin",
        how="left"
    )
)

# Add CF to the info table

cf_pdf = df_cf_pd[["vin", "CF"]]
cf_spark = spark.createDataFrame(cf_pdf)

info_final = info_extended.join(cf_spark, on="vin", how="left")

# 11.	PUSH ESSENTIAL DATA TO CATALOG
info_final.write.mode("overwrite").saveAsTable("pt_configuration_thesis.alltrucks_v1.vin_info_sg_info_cluster_info_talpy_info_MDS_info_CF_info")

# 12.	POSTPROCESSING - PLOTS - FOR SELECTED SALES GROUP
# Merge sales group info into df_cf_pd
df_cf_sg = df_cf_pd.merge(
    info.select("vin", "sales_group").toPandas(),
    on="vin",
    how="inner"
)

cf_bin_width = 0.05
cf_bins = np.arange(0, 1 + cf_bin_width, cf_bin_width)  # 0 to 1
df_cf_sg['CF_bin_fine'] = pd.cut(df_cf_sg['CF'], bins=cf_bins, include_lowest=True)

def get_histogram(sales_group=1):
    # Filter for selected sales group
    df_sg = df_cf_sg[df_cf_sg['sales_group'] == sales_group]
    
    # Count trucks per CF bin
    counts = df_sg.groupby('CF_bin_fine')['vin'].nunique()
    
    # Total trucks in this sales group
    total_trucks = df_sg['vin'].nunique()
    
    return counts, total_trucks


import matplotlib.pyplot as plt

sg = 1  # choose 1 to 15
counts, total_trucks = get_histogram(sg)

plt.figure(figsize=(12,6))
plt.bar(range(len(counts)), counts.values, width=1.0, edgecolor='k', alpha=0.7, color='red')
plt.xticks(range(len(counts)), [f"{interval.left:.2f}-{interval.right:.2f}" for interval in counts.index], rotation=45)
plt.xlabel("Challenge Factor (CF) Interval")
plt.ylabel("Number of Trucks")
plt.title(f"Truck Distribution Across CF Bins — SG{sg} (Total Trucks: {total_trucks})")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sg = 1  # choose 1 to 15
counts, total_trucks = get_histogram(sg)

# Convert interval bins to midpoints
x = np.array([
    (interval.left + interval.right) / 2
    for interval in counts.index
])

y = counts.values

# Smooth counts (controls curve smoothness)
y_smooth = gaussian_filter1d(y, sigma=1.2)

plt.figure(figsize=(12, 6))

# Plot smoothed curve
plt.plot(x, y_smooth, color='black', linewidth=2)

# Shade area under curve
plt.fill_between(x, y_smooth, color='red', alpha=0.3)

# Axis labels (big + bold)
plt.xlabel("Challenge Factor (-)", fontsize=15, fontweight='bold')
plt.ylabel("Number of Trucks (-)", fontsize=15, fontweight='bold')

plt.title(f"Truck Distribution Across CF — SG{sg} (Total Trucks: {total_trucks})", fontsize=15, fontweight='bold')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
PLOT - NUMBER OF TRUCKS OF ALL GROUPS

import numpy as np
import pandas as pd

# Define CF bin width
cf_bin_width = 0.05
cf_bins = np.arange(0, 1 + cf_bin_width, cf_bin_width)
cf_labels = [f"{round(left,2)}-{round(right,2)}" for left, right in zip(cf_bins[:-1], cf_bins[1:])]

# Prepare a dictionary to store normalized distributions per sales group
sg_distributions = {}

for sg in sorted(df_cf_sg['sales_group'].unique()):
    df_sg = df_cf_sg[df_cf_sg['sales_group'] == sg]
    
    # Count trucks per bin
    counts, _ = np.histogram(df_sg['CF'], bins=cf_bins)
    
    # Normalize to 0-1
    counts_norm = counts / counts.max() if counts.max() > 0 else counts
    sg_distributions[sg] = counts_norm

import matplotlib.pyplot as plt

n_rows, n_cols = 5, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20), sharex=True, sharey=True)
axes = axes.flatten()  

for i, sg in enumerate(sorted(df_cf_sg['sales_group'].unique())):
    ax = axes[i]
    
    counts_norm = sg_distributions[sg]
    total_trucks = df_cf_sg[df_cf_sg['sales_group']==sg]['vin'].nunique()
    
    ax.bar(range(len(counts_norm)), counts_norm, width=1.0, edgecolor='k', alpha=0.7)
    ax.set_title(f"SG{sg} (Total: {total_trucks})", fontsize=10)
    
    # Optional: x-axis labels only on bottom row
    if i >= (n_rows-1)*n_cols:
        ax.set_xticks(range(len(cf_labels)))
        ax.set_xticklabels(cf_labels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    # Optional: y-axis labels only on left column
    if i % n_cols == 0:
        ax.set_ylabel("Normalized Trucks")


plt.suptitle("Normalized Truck Distributions Across CF — All Sales Groups", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Step 1: Prepare normalized CF distributions
# -----------------------------
cf_bin_width = 0.05
cf_bins = np.arange(0, 1 + cf_bin_width, cf_bin_width)
cf_bin_centers = (cf_bins[:-1] + cf_bins[1:]) / 2  # midpoints for plotting, alternatively can be craeted using bins

sg_distributions = {}
for sg in sorted(df_cf_sg['sales_group'].unique()):
    df_sg = df_cf_sg[df_cf_sg['sales_group'] == sg]
    
    counts, _ = np.histogram(df_sg['CF'], bins=cf_bins)
    counts_norm = counts / counts.max() if counts.max() > 0 else counts
    sg_distributions[sg] = counts_norm

# -----------------------------
# Step 2: Create grid of subplots
# -----------------------------
n_rows, n_cols = 5, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 20), sharey=True)
axes = axes.flatten()

# -----------------------------
# Step 3: Plot curves for each sales group
# -----------------------------
for i, sg in enumerate(sorted(df_cf_sg['sales_group'].unique())):
    ax = axes[i]
    
    counts_norm = sg_distributions[sg]
    total_trucks = df_cf_sg[df_cf_sg['sales_group'] == sg]['vin'].nunique()
    
    # Plot curve and optional shaded area
    ax.plot(cf_bin_centers, counts_norm, linewidth=2, color='darkblue')
    ax.fill_between(cf_bin_centers, 0, counts_norm, color='darkblue', alpha=0.2)
    
    # Set title with total trucks
    ax.set_title(f"SG{sg} (Total: {total_trucks})", fontsize=10)
    
    # X-axis labels for all plots
    ax.set_xticks(cf_bin_centers)
    ax.set_xticklabels([f"{round(c,2)}" for c in cf_bin_centers], rotation=90, fontsize=8)
    
    # Y-axis labels only for left column
    if i % n_cols == 0:
        ax.set_ylabel("Normalized Trucks")

# -----------------------------
# Step 4: Layout adjustments
# -----------------------------
plt.suptitle("Normalized CF Distributions — All Sales Groups", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

