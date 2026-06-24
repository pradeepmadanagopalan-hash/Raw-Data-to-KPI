'''
0. README
Notebook: MDS_Gradient
Inputs
•	pt_configuration_thesis.alltrucks_v1.combined_gradient_spectra_outlier_removed
•	pt_configuration_thesis.alltrucks_v1.weight_spectra_outliers
Outputs
•	pt_configuration_thesis.alltrucks_v1.gradient_MDS_results
Purpose
•	This notebook is meant to take as input the combined gradient spectra from Spectra_Outlier_Removal and weight_spectra_outliers from Weight_Spectra_Outlier_Detection.
•	Then a similar data preparation process as previous notebooks is used, wasserstein distance matrix is generated and then the matrix is downscaled using the technique 'Multi Dimensional Scaling' (19481X19481 to 19481 X3). The downscaled matrix is written to the catalog.
Other remarks
•	The notebook has some pro processing metrics and plots like Kruskal Stress and Shepard Diagram
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
df_gradient_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`combined_gradient_spectra_outlier_removed`")

unique_vins_count_ori = df_gradient_spectra .select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

df_outliers = spark.table("pt_configuration_thesis.alltrucks_v1.`weight_spectra_outliers`")

unique_vins_count_ori = df_outliers.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

outlier_vins = df_outliers.select("vin").distinct()
df_gradient_cleaned = df_gradient_spectra.join(outlier_vins, on="vin", how="left_anti")

unique_vins_count_ori = df_gradient_cleaned .select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

# 3.	MAKE DATA READY - SAME PROCESS AS WHAT WAS FOLLOWED IN THE NOTEBOOK WASSERSTEIN_DISTANCE_GRADIENT_SPECTRA
# STEP 1 - JOIN VECTORS 
from pyspark.sql.functions import lit

df_gradient_all = df_gradient_cleaned

display(df_gradient_all)

# STEP 2 - COMPUTE BIN CENTER
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

def get_bin_center(interval_str):
    cleaned = interval_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    left, right = cleaned.split(",")
    return (float(left.strip()) + float(right.strip())) / 2

get_bin_center_udf = udf(get_bin_center, DoubleType())

df_gradient_all = df_gradient_all.withColumn("bin_center", get_bin_center_udf("x1_interval"))

display(df_gradient_all)

# STEP 3 - COMPUTE GLOBAL MAXIMA
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min

global_max_gradient = df_gradient_all.agg(spark_max("bin_center")).collect()[0][0]
print(f"Global max gradient across all trucks: {global_max_gradient}")

global_min_gradient = df_gradient_all.agg(spark_min("bin_center")).collect()[0][0]
print(f"Global min gradient across all trucks: {global_min_gradient}")

# STEP 4 - GENERATE ALL BIN CENTERS DEPENDING ON GLOBAL MAXIMA
import numpy as np

bin_interval = 1
# all_bins = np.arange(0 + bin_interval/2, global_max_velocity + bin_interval, bin_interval)

# all_bins = np.arange(12000 + bin_interval/2, 49000 + bin_interval, bin_interval)

# all_bins = np.arange(global_min_gradient + bin_interval, global_max_gradient + bin_interval, bin_interval)

# all_bins = np.arange(-7.5 + bin_interval, 6.5 + bin_interval, bin_interval)

all_bins = np.arange(-0.5 + bin_interval, 9.5 + bin_interval, bin_interval)


print(f"All bin centers: {all_bins}")

# -------------------------------
# STEP 5 - Add missing bins for each truck (gradient spectra)
# -------------------------------

# Convert Spark DataFrame → Pandas
df_pd = df_gradient_all.select("vin", "bin_center", "count").toPandas()

# Pivot to wide form: rows=trucks, columns=bin centers
df_truck_distributions = df_pd.pivot_table(
    index="vin",
    columns="bin_center",
    values="count",
    aggfunc="sum",
    fill_value=0
)

# Ensure all bins exist (fill missing bins with 0)
df_truck_distributions = df_truck_distributions.reindex(columns=all_bins, fill_value=0)

# Convert counts → probability distribution
X = df_truck_distributions.to_numpy(dtype=float)
row_sums = X.sum(axis=1, keepdims=True)
X_probs = X / row_sums

# Extract bin centers and VINs
bin_centers = df_truck_distributions.columns.values.astype(float)
vins = df_truck_distributions.index.values

# Print shape of probability matrix
print("X_probs shape:", X_probs.shape)

import numpy as np

# -------------------------------
# Identify all-zero rows using raw counts X
# -------------------------------
row_sums_before = X.sum(axis=1)  # sum across bins for each truck
zero_rows = np.where(row_sums_before == 0)[0]  # indices of all-zero rows

# Display rows that will be removed
print("Rows removed (all-zero distributions):", zero_rows)

# Total number of rows removed
print("Total rows removed:", len(zero_rows))

# Remove all-zero rows from X, X_probs, vins, and df_truck_distributions
X = np.delete(X, zero_rows, axis=0)
vins = np.delete(vins, zero_rows, axis=0)
df_truck_distributions = df_truck_distributions.drop(df_truck_distributions.index[zero_rows])

# Recompute probability distributions for remaining trucks
row_sums = X.sum(axis=1, keepdims=True)
X_probs = X / row_sums

# Display new shape
print("New X_probs shape:", X_probs.shape)

# 4.	WASSERSTEIN DISTANCE CALCULATION – NEW
import numpy as np
from numba import njit, prange
import time

# -------------------------------
# CONFIGURATION
# -------------------------------
n_trucks = X_probs.shape[0]
bin_centers_array = bin_centers.astype(np.float64)
block_size = 1000  # block wise wasserstein distance to improve speed of the code
# -------------------------------
# NUMBA FUNCTION: 1D Wasserstein Distance Calculation 
# Numba JIT is used to reduce computational overhead in large-scale pairwise distance calculations
# I have tried my best to reduce several hours of computation to a few seconds. Maybe this can be further optimzized
# -------------------------------
@njit
def wasserstein_1d(u, v, bin_centers):
    """
    Exact 1D Wasserstein distance (EMD) between two discrete distributions u, v
    with the same support given by bin_centers
    """
    cdf_u = np.cumsum(u)
    cdf_v = np.cumsum(v)
    distance = np.sum(np.abs(cdf_u - cdf_v) * np.diff(np.append(bin_centers, bin_centers[-1]+1)))
    return distance

# -------------------------------
# BLOCK-WISE DISTANCE MATRIX CALCULATION
# -------------------------------
distance_matrix = np.zeros((n_trucks, n_trucks), dtype=np.float64)

start_time = time.time()
print(f"Computing Wasserstein distance matrix for {n_trucks} trucks in blocks of {block_size}...")

# Precompute bin widths for efficiency
bin_widths = np.diff(np.append(bin_centers_array, bin_centers_array[-1]+1))

# Numba JIT for block computation
@njit(parallel=True)
def compute_block(X, start_i, end_i, distance_matrix, bin_widths):
    n = X.shape[0]
    for i in prange(start_i, end_i):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.sum(np.abs(np.cumsum(X[i]) - np.cumsum(X[j])) * bin_widths)
            distance_matrix[j, i] = distance_matrix[i, j]

# Loop over blocks
for start in range(0, n_trucks, block_size):
    end = min(start + block_size, n_trucks)
    compute_block(X_probs, start, end, distance_matrix, bin_widths)
    elapsed = time.time() - start_time
    print(f"Processed trucks {start} to {end} | Elapsed: {elapsed:.2f} s")

total_time = time.time() - start_time
print(f"Finished computing Wasserstein distance matrix in {total_time/60:.2f} minutes")

# Tip - please check if you have 1. only zeroes along diagonal 2. symmetric numbers about diagonal 3. Min distance should be zero and never negative 4. Shape of matrix should be nXn where n is the number of trucks we considered
print("Distance matrix shape:", distance_matrix.shape)
print("Min distance:", distance_matrix.min())
print("Max distance:", distance_matrix.max())
print("Example distances (first 5 trucks):\n", distance_matrix[:5, :5])

# 5.	MULTI DIMENSIONAL SCALING
from sklearn.manifold import MDS
import numpy as np
import pandas as pd

# Assuming `distance_matrix` is a PySpark DataFrame 
D = distance_matrix  

# Initialize MDS
mds = MDS(
    n_components=3,           # 2 or 3 is typical, , here 3 because we are downscaling to 3D dimension
    dissimilarity='precomputed',
    random_state=42,
    n_init=4,                 # multiple restarts for stability
    max_iter=300
)

# Note that the hyperparameters can be tuned

# Fit and transform
X_w = mds.fit_transform(D)    

# Wrap in a DataFrame for convenience
trucks = [f"T{i+1}" for i in range(X_w.shape[0])]
embedding_df = pd.DataFrame(X_w, columns=["MDS1_g", "MDS2_g", "MDS3_g"], index=trucks)

print(embedding_df.head())

display(embedding_df)

# 6.	MERGE BACK WITH VIN AND SALES GROUP INFO
# Use actual VINs as index
embedding_df = pd.DataFrame(X_w, columns=["MDS1_g", "MDS2_g", "MDS3_g"], index=vins)

# Convert Spark DataFrame to Pandas with vin and sales_group
df_vin_sales = df_gradient_all.select("vin", "sales_group").dropDuplicates().toPandas()
df_vin_sales = df_vin_sales.set_index("vin")

# Join sales_group to embedding_df
embedding_df = embedding_df.join(df_vin_sales, how="left")

# Reset index to make 'vin' a column
embedding_df = embedding_df.reset_index().rename(columns={"index": "vin"})

# Reorder columns
embedding_df = embedding_df[["vin", "sales_group", "MDS1_g", "MDS2_g", "MDS3_g"]]

display(embedding_df)


# Just for verification before pushing to catalog
# Count number of VINs per sales group
sales_group_counts = embedding_df.groupby("sales_group")["vin"].count().reset_index()
sales_group_counts = sales_group_counts.rename(columns={"vin": "num_vins"})
sales_group_counts = sales_group_counts.sort_values("sales_group")

display(sales_group_counts)

# 7.	PUSH ESSENTIAL DATA TO CATALOG
# Convert Pandas DataFrame to Spark DataFrame
spark_embedding_df = spark.createDataFrame(embedding_df)

# Save as a Hive/Delta table
spark_embedding_df.write.mode("overwrite").saveAsTable(
    "pt_configuration_thesis.alltrucks_v1.gradient_MDS_results"
)

# 8.	POST PROCESSING - KRUSKAL STRESS VALUES
print(f"Final stress: {mds.stress_:.3f}")

# Original dissimilarities (flattened)
sum_d2 = np.sum(D ** 2) / 2  # divide by 2 since matrix is symmetric
normalized_stress = np.sqrt(mds.stress_ / sum_d2)
print(f"Normalized Stress: {normalized_stress:.4f}")

# 9.	POST PROCESSING - SHEPARD DIAGRAM

import matplotlib.pyplot as plt

plt.scatter(embedding_df["MDS1_g"], embedding_df["MDS2_g"])
plt.xlabel("MDS1_g"); plt.ylabel("MDS2_g")
plt.title("MDS embedding of Gradient Spectra (Wasserstein geometry)")
plt.show()

#Quality Check
from sklearn.metrics import pairwise_distances
D_embedded = pairwise_distances(X_w)
corr = np.corrcoef(D.flatten(), D_embedded.flatten())[0, 1]
print(f"Correlation between original and embedded distances: {corr:.3f}")


