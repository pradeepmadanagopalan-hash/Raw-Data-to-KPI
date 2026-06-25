'''
0. README
Notebook: Prepare_Spectra_and_join_SFC
Inputs
•	pt_configuration_thesis.alltrucks_v1.engine_spectra_all_cleaned
•	pt_configuration_thesis.alltrucks_v1.tco_fuel_map_50_50
•	pt_configuration_thesis.alltrucks_v1.hp_fuel_map_50_50
•	pt_configuration_thesis.alltrucks_v1.sales_group_mapping
Outputs
•	pt_configuration_thesis.alltrucks_v1.engine_spectra_all_cleaned_with_SFC
Purpose
•	This notebook is meant to take as input the combined engine spectra data of all vins (from notebook Engine_Spectra_Outlier_Removal) and also the interpolated fuel map data (i.e. SFC values) (from the notebook Engine_Map_TCO&HP_Generation).
•	It filters the spectra for the RPM X Torque range for which we intend to do our LEMP calculations and then for each grid (50 RPM x 50 Nm) in the engine map - for each vin, it adds the corresponding SFC value.
•	So at the end of this step, the engine spectra additionally has the SFC value associated with each grid. Earlier it had only the mileage.
•	This addition of SFC values has happend for both HP and TCO maps depending on which sales group has which engine power rating. This is done manually now, can be automated later if needed.
•	engine_spectra_all_cleaned_with_SFC is the dataframe ready for LEMP calculation
Other remarks
•	Honestly, I feel like I have over-complicated the SFC addition steps in this notebook I bit. It solves the purpose, no doubt - but I feel the code can be streamlined and made a bit simpler.
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
# Load table and rename column
df_sg_engine_map_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`engine_spectra_all_cleaned`")
unique_vins_count_ori = df_sg_engine_map_spectra.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

display(df_sg_engine_map_spectra)

# Input Interpolated fuel map Data
fuel_map_tco = spark.table("pt_configuration_thesis.alltrucks_v1.tco_fuel_map_50_50")

display(fuel_map_tco)

# Input Interpolated fuel map Data
fuel_map_hp = spark.table("pt_configuration_thesis.alltrucks_v1.hp_fuel_map_50_50")

display(fuel_map_hp)

# 3.	EXTRACT ONLY DISTANCE SPECTRA
# Extract only Distance Spectra
# Filter rows where ls_config_id == 2
df_mileage_spectra = df_sg_engine_map_spectra.filter(f.col("ls_config_id") == 2)

display(df_mileage_spectra)

# Count rows in the filtered engine spectra
num_rows_spectra = df_mileage_spectra.count()
print(f"Number of rows in df_mileage_spectra: {num_rows_spectra}")

# Example: count rows in the raw data before filtering 
num_rows_spectra_before = df_sg_engine_map_spectra.count()
print(f"Number of rows in df_sg_engine_map_spectra: {num_rows_spectra_before}")

# 4.	FILTER FOR SPECIFIC TORQUE RANGE AND RPM RANGE
# Filter FOR TORQUE (250 Nm +) AND RPM RANGE (850 - 2000)
# For peak torque we set physical limits
from pyspark.sql import functions as F

from pyspark.sql import functions as F
'''
df_filtered = df_mileage_spectra.withColumn(
    "max_allowed_torque",
    F.when(F.col("engine_power_rating_sales_code").isin("M3A"), 2100)
     .when(F.col("engine_power_rating_sales_code").isin("M3B"), 2200)
     .when(F.col("engine_power_rating_sales_code").isin("M3C"), 2300)
     .when(F.col("engine_power_rating_sales_code").isin("M3D"), 2500)
     .when(F.col("engine_power_rating_sales_code").isin("M3E"), 2600)
)
'''

# Filter for RPM between 850 and 1950, and Torque higher than 250 Nm
df_filtered = df_mileage_spectra.filter(
    (F.col("x1_quantized") >= 850) &
    (F.col("x1_quantized") <= 1950) &
    (F.col("x2_quantized") >= 250) &
    (F.col("x2_quantized") <= 2550)
)

display(df_filtered)

# Count rows in the filtered engine spectra
num_rows_spectra = df_mileage_spectra.count()
print(f"Number of rows in df_mileage_spectra: {num_rows_spectra}")

# Count rows in the filtered dataframe 
num_rows_filtered = df_filtered.count()
print(f"Number of rows in df_filtered: {num_rows_filtered}")

# 5.	MAKE EQUAL NUMBER OF ROWS FOR ALL TRUCKS
# Make Equal number of rows for all Trucks
from pyspark.sql import functions as F
from pyspark.sql import Window

# STEP 1: Find global maximum torque across all trucks
global_torque_max = df_filtered.agg(F.max("x2_quantized")).collect()[0][0]
print(f"Global max torque = {global_torque_max} Nm")

# STEP 2: Create full torque grid (0, 100, 200, ..., global_torque_max)
torque_grid = list(range(250, int(global_torque_max) + 50, 50))
torque_df = spark.createDataFrame([(t,) for t in torque_grid], ["x2_quantized"])

# STEP 3: Get all unique (vin, RPM) combinations
rpm_vin_df = df_filtered.select(
    "vin", "sales_group", "x1_quantized", "x1_interval", "x1_value_to_text", "x1_signal_id"
).distinct()

# STEP 4: Cross join VIN-RPM grid with full torque bins
full_grid = rpm_vin_df.crossJoin(torque_df)

# STEP 5: Add derived columns for torque bins and defaults
full_grid = (
    full_grid
    .withColumn(
        "x2_interval",
        F.concat(
            F.lit("["), F.col("x2_quantized"), F.lit(","), (F.col("x2_quantized") + 50), F.lit(")")
        )
    )
    .withColumn("x2_value_to_text", F.col("x2_interval"))
    .withColumn("ls_config_id", F.lit(2))
)

# STEP 6: Attach correct signal IDs dynamically
signal_ids = df_filtered.select("vin", "x2_signal_id").distinct()
full_grid = full_grid.join(signal_ids, on="vin", how="left")

# STEP 7: Add count = 0 for missing bins
full_grid = full_grid.withColumn("count", F.lit(0.0))

# STEP 8: Join back with original df to replace counts where available
df_filtered_renamed = df_filtered.select(
    "vin", "x1_quantized", "x2_quantized", F.col("count").alias("count_actual")
)

df_filled = (
    full_grid.join(
        df_filtered_renamed,
        on=["vin", "x1_quantized", "x2_quantized"],
        how="left"
    )
    .withColumn("count", F.coalesce(F.col("count_actual"), F.col("count")))
    .drop("count_actual")
)

# df_filled now contains a complete and consistent grid for all trucks  --> Again this is necessary for LEMP calculation in the next notebooks
df_filled.show(10)
print(f"Final row count: {df_filled.count():,}")

display(df_filled)

# Section just for verification

from pyspark.sql import functions as F

# Count distinct VINs per sales group
vin_counts_per_sg = df_filled.groupBy("sales_group") \
    .agg(F.countDistinct("vin").alias("distinct_vins")) \
    .orderBy("sales_group")

vin_counts_per_sg.show()

# 6.	NORMALIZE THE MILEAGE
# Normalize Mileage
from pyspark.sql import Window
from pyspark.sql import functions as F

# Step 1: Compute total mileage per truck
window_vin = Window.partitionBy("vin")
df_norm = df_filled.withColumn(
    "total_mileage",
    F.sum("count").over(window_vin)
)

# Step 2: Compute fractional mileage
df_norm = df_norm.withColumn(
    "fraction_mileage",
    F.when(F.col("total_mileage") > 0, F.col("count") / F.col("total_mileage")).otherwise(0)
)

# Step 3: Drop helper column if not needed
df_norm = df_norm.drop("total_mileage")

# Each truck now has a normalized mileage distribution on engine map
df_norm.show(10)

display(df_norm)

# Now we check if all VINs have a sum of fraction_mileage equal to 1 (Verification step to see if the normalization works correctly)

from pyspark.sql import functions as F
from pyspark.sql import Window

# Window per VIN
window_vin = Window.partitionBy("vin")

# Sum fraction_mileage per VIN
df_check = df_norm.withColumn(
    "sum_fraction",
    F.sum("fraction_mileage").over(window_vin)
).select("vin", "sum_fraction").distinct()

# Check if all sums are close to 1
# Allowing small floating point tolerance
tolerance = 1e-8
all_ok = df_check.filter(F.abs(F.col("sum_fraction") - 1) > tolerance).count() == 0

# Print message
if all_ok:
    print("All VINs satisfy the condition: fraction_mileage sums to 1.")
else:
    print("Some VINs do NOT satisfy the condition: fraction_mileage does not sum to 1.")

# 7.	QUICK VIZ HEATMAP CHECK
# Again this is a verification step if all the steps done so far are working correctly. 
# We can do this by quickly having a look at the engine operation heatmap for any random vin

import matplotlib.pyplot as plt
import pandas as pd

# Choose a VIN to visualize
vin_to_plot = "0194f39f2ddefcf31cb44b2e684302ee2154a828f4186ecaed93861d1ea12650"  # example (you can select any other vin also)

# Step 1: Filter one truck
df_vin = df_norm.filter(F.col("vin") == vin_to_plot)

# Step 2: Convert to Pandas for plotting
pdf = df_vin.select("x1_quantized", "x2_quantized", "fraction_mileage").toPandas()

# Step 3: Pivot the data to make a grid
pivot_df = pdf.pivot_table(
    index="x2_quantized",      # torque on Y-axis
    columns="x1_quantized",    # RPM on X-axis
    values="fraction_mileage",
    fill_value=0
)

# Step 4: Plot the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(
    pivot_df,
    aspect="auto",
    origin="lower",  # lower torque at bottom
    cmap="YlOrRd"
)
plt.colorbar(label="Mileage Fraction")
plt.title(f"Truck VIN: {vin_to_plot}\nMileage Fraction Heatmap (RPM vs Torque)")
plt.xlabel("Engine Speed (RPM)")
plt.ylabel("Torque (Nm)")

# X & Y ticks — use bin centers
plt.xticks(
    range(len(pivot_df.columns)),
    pivot_df.columns,
    rotation=45
)
plt.yticks(
    range(len(pivot_df.index)),
    pivot_df.index
)

plt.tight_layout()
plt.show()

# 8.	TRIM FUEL MAP DATA – TCO
from pyspark.sql import functions as F
from pyspark.sql import Window

# Trim Map Data

# Step 1: Add a row index (since PySpark DataFrames are unordered)
w = Window.orderBy(F.monotonically_increasing_id())
fuel_map_indexed_tco = fuel_map_tco.withColumn("row_index", F.row_number().over(w) - 1)

# Step 2: Compute RPM bin lower bound based on row index
# Each row represents a 50 RPM interval
fuel_map_indexed_tco = fuel_map_indexed_tco.withColumn("rpm_bin", F.col("row_index") * 50)

# Step 3: Filter to only rows with RPM between 850 and 2000 (inclusive)
fuel_map_trimmed_tco = fuel_map_indexed_tco.filter((F.col("rpm_bin") >= 850) & (F.col("rpm_bin") <= 2000))

# Step 4: Drop torque columns below 250 Nm
columns_to_drop = ['0_50', '50_100', '100_150', '150_200', '200_250']
fuel_map_trimmed_tco = fuel_map_trimmed_tco.drop(*columns_to_drop)


# Step 5: Optional — drop helper columns or keep rpm_bin
fuel_map_trimmed_tco = fuel_map_trimmed_tco.drop("row_index")

# Check result
fuel_map_trimmed_tco.select("rpm_bin").show(5)
print(f"Rows before: {fuel_map_tco.count()}, after trimming: {fuel_map_trimmed_tco.count()}")


display(fuel_map_trimmed_tco)

# 9.	FUEL MAP DATA IN LONG FORMAT – TCO
# Convert map data to long format
from pyspark.sql import functions as F
from pyspark.sql import Window

# Add row index to determine rpm_bin (since Spark has no implicit order)
w = Window.orderBy(F.monotonically_increasing_id())
fuel_map_indexed_tco = fuel_map_tco.withColumn("row_index", F.row_number().over(w) - 1)

# Compute rpm_bin (each row = 50 RPM)
fuel_map_indexed_tco = fuel_map_indexed_tco.withColumn("rpm_bin", F.col("row_index") * 50)

# Filter for 850–2000 RPM
fuel_map_trimmed_tco = fuel_map_indexed_tco.filter((F.col("rpm_bin") >= 850) & (F.col("rpm_bin") <= 2000))
columns_to_drop = ['0_50', '50_100', '100_150', '150_200', '200_250']
fuel_map_trimmed_tco = fuel_map_trimmed_tco.drop(*columns_to_drop)

# Identify torque columns dynamically
torque_columns = [c for c in fuel_map_trimmed_tco.columns if "_" in c and c not in ["row_index", "rpm_bin"]]

# Build the F.stack() expression safely
stack_expr = F.expr(
    "stack({}, {})".format(
        len(torque_columns),
        ", ".join([f"'{c}', `{c}`" for c in torque_columns])
    )
)

# Apply unpivot (wide → long)
fuel_map_long_tco = fuel_map_trimmed_tco.select("rpm_bin", stack_expr.alias("torque_bin", "SFC"))

# Extract numeric lower bound from torque_bin (e.g., '200_300' → 200)
fuel_map_long_tco = fuel_map_long_tco.withColumn(
    "x2_quantized",
    F.split("torque_bin", "_").getItem(0).cast("int")
)

# Drop null SFC values
fuel_map_long_tco = fuel_map_long_tco.filter(F.col("SFC").isNotNull())

# Check result
fuel_map_long_tco.show(10)


display(fuel_map_long_tco)

# 10.	TRIM FUEL MAP DATA – HP
from pyspark.sql import functions as F
from pyspark.sql import Window

# Trim Map Data

# Step 1: Add a row index (since PySpark DataFrames are unordered)
w = Window.orderBy(F.monotonically_increasing_id())
fuel_map_indexed_hp = fuel_map_hp.withColumn("row_index", F.row_number().over(w) - 1)

# Step 2: Compute RPM bin lower bound based on row index
# Each row represents a 50 RPM interval
fuel_map_indexed_hp = fuel_map_indexed_hp.withColumn("rpm_bin", F.col("row_index") * 50)

# Step 3: Filter to only rows with RPM between 850 and 2000 (inclusive)
fuel_map_trimmed_hp = fuel_map_indexed_hp.filter((F.col("rpm_bin") >= 850) & (F.col("rpm_bin") <= 2000))

# Step 4: Drop torque columns below 250 Nm
columns_to_drop = ['0_50', '50_100', '100_150', '150_200', '200_250']
fuel_map_trimmed_hp = fuel_map_trimmed_hp.drop(*columns_to_drop)


# Step 5: Optional — drop helper columns or keep rpm_bin
fuel_map_trimmed_hp = fuel_map_trimmed_hp.drop("row_index")

# Check result
fuel_map_trimmed_hp.select("rpm_bin").show(5)
print(f"Rows before: {fuel_map_hp.count()}, after trimming: {fuel_map_trimmed_hp.count()}")


display(fuel_map_trimmed_hp)

# 11.	FUEL MAP DATA IN LONG FORMAT – HP
# Convert map data to long format
from pyspark.sql import functions as F
from pyspark.sql import Window

# Add row index to determine rpm_bin (since Spark has no implicit order)
w = Window.orderBy(F.monotonically_increasing_id())
fuel_map_indexed_hp = fuel_map_hp.withColumn("row_index", F.row_number().over(w) - 1)

# Compute rpm_bin (each row = 50 RPM)
fuel_map_indexed_hp = fuel_map_indexed_hp.withColumn("rpm_bin", F.col("row_index") * 50)

# Filter for 850–2000 RPM
fuel_map_trimmed_hp = fuel_map_indexed_hp.filter((F.col("rpm_bin") >= 850) & (F.col("rpm_bin") <= 2000))
columns_to_drop = ['0_50', '50_100', '100_150', '150_200', '200_250']
fuel_map_trimmed_hp = fuel_map_trimmed_hp.drop(*columns_to_drop)

# Identify torque columns dynamically
torque_columns = [c for c in fuel_map_trimmed_hp.columns if "_" in c and c not in ["row_index", "rpm_bin"]]

# Build the F.stack() expression safely
stack_expr = F.expr(
    "stack({}, {})".format(
        len(torque_columns),
        ", ".join([f"'{c}', `{c}`" for c in torque_columns])
    )
)

# Apply unpivot (wide → long)
fuel_map_long_hp = fuel_map_trimmed_hp.select("rpm_bin", stack_expr.alias("torque_bin", "SFC"))

# Extract numeric lower bound from torque_bin (e.g., '200_300' → 200)
fuel_map_long_hp = fuel_map_long_hp.withColumn(
    "x2_quantized",
    F.split("torque_bin", "_").getItem(0).cast("int")
)

# Drop null SFC values
fuel_map_long_hp = fuel_map_long_hp.filter(F.col("SFC").isNotNull())

# Check result
fuel_map_long_hp.show(10)

display(fuel_map_long_hp)

# 12.	SEE WHICH SALES GROUPS HAVE TCO MAP AND WHICH HAVE HP MAP
# Load table and rename column
sales_group_map = spark.table("pt_configuration_thesis.alltrucks_v1.`sales_group_mapping`")
unique_vins_count_ori = sales_group_map.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

from pyspark.sql import functions as F

# Keep only distinct sales_group to engine_description combinations
sales_group_engine_map = sales_group_map.select(
    "sales_group", "engine_description"
).distinct()

display(sales_group_engine_map)


# 13.	COMBINE MILEAGE DATA WITH CORRECT SFC DATA
# Rename TCO map
fuel_map_long_tco_renamed = fuel_map_long_tco.select(
    F.col("rpm_bin").alias("x1_quantized"),
    F.col("x2_quantized"),
    F.col("SFC")
)

# Rename HP map
fuel_map_long_hp_renamed = fuel_map_long_hp.select(
    F.col("rpm_bin").alias("x1_quantized"),
    F.col("x2_quantized"),
    F.col("SFC")
)
# Filter mileage data for TCO sales groups
# Note that I have done this manually now. This can ofcourse be changes to automatically detect which SG has TCO engine and which SG has HP engine and join data according to that

tco_groups = [1, 2, 5, 6, 7, 8, 10, 12, 13, 14, 15]
hp_groups  = [3, 4, 9, 11]


df_tco = df_norm.filter(F.col("sales_group").isin(tco_groups))

# Join with TCO fuel map
df_tco_joined = df_tco.join(
    fuel_map_long_tco_renamed,
    on=["x1_quantized", "x2_quantized"],
    how="left"
)

# Filter mileage data for HP sales groups
df_hp = df_norm.filter(F.col("sales_group").isin(hp_groups))

# Join with HP fuel map
df_hp_joined = df_hp.join(
    fuel_map_long_hp_renamed,
    on=["x1_quantized", "x2_quantized"],
    how="left"
)
df_combined = df_tco_joined.unionByName(df_hp_joined)
display(df_combined)
from pyspark.sql import functions as F

df_combined = df_combined.withColumn(
    "engine_map",
    F.when(F.col("sales_group").isin(tco_groups), F.lit("TCO"))
     .when(F.col("sales_group").isin(hp_groups), F.lit("HP"))
     .otherwise(F.lit("UNKNOWN"))  # safety catch if any sales_group is missing
)

# Optional: check
df_combined.select("sales_group", "engine_map").distinct().show()
display(df_combined)
# Final Data before calculating penalties
df_final = df_combined.select(
    "vin",
    "sales_group",
    "engine_map",
    "x1_interval",
    "x2_interval",
    "count",
    "fraction_mileage",
    "SFC"
)

# Optional: inspect top rows
df_final.show(10)
display(df_final)


# 14.	WRITE DATA TO THE CATALOG

df_final.write.mode("overwrite").saveAsTable(
    "pt_configuration_thesis.alltrucks_v1.engine_spectra_all_cleaned_with_SFC"
)
