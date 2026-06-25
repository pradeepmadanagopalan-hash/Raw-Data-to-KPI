'''
0. README
Notebook: Engine_Spectra_Outlier_Removal
Inputs
•	pt_configuration_thesis.alltrucks_v1.sgX_enginespectra (X = 1,2,....15)
•	pt_configuration_thesis.alltrucks_v1.vin_info_fin_info_sgmap_info_cluster_info_talpy_info_mds_info_cf_info
Outputs
•	pt_configuration_thesis.alltrucks_v1.engine_spectra_all_cleaned
Purpose
•	This notebook is meant to take as input the engine spectras generated from the notebook Load_spectra_heatmap_3d_definition and remove of the spectras of all the outlier vins.
•	The output of this notebook is a combined dataframe having the engine spectra of all the non-outlier vins which can in subsequent notebooks be used for creating the LEMP KPI.
Other remarks
•	None
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
# Load table
df_sg1_engine_map_spectra = spark.table("pt_configuration_thesis.alltrucks_v1.`sg1_enginespectra`")
unique_vins_count_ori = df_sg1_engine_map_spectra.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

info = spark.table("pt_configuration_thesis.alltrucks_v1.`vin_info_fin_info_sgmap_info_cluster_info_talpy_info_mds_info_cf_info`")
unique_vins_count_ori = info.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

# 3.	PREPARE VIN REFERENCE
# Done to join metadata back in the last step

from pyspark.sql import functions as f

info_vins = (
    info
    .select("vin", "sales_group")
    .distinct()
)

# 4.	ITERATE ALL 15 ENGINE SPECTRA TABLES
# This section iterates over the engine spectra of all 15 sales groups and gets rid of all the outliers identified in the previous notebooks.

cleaned_spectra_dfs = []
verification_rows = []

for sg in range(1, 16):

    table_name = f"pt_configuration_thesis.alltrucks_v1.`sg{sg}_enginespectra`"
    
    # Load raw spectra
    df = spark.table(table_name)
    
    # Add sales_group column
    df = df.withColumn("sales_group", f.lit(sg))
    
    # VIN count BEFORE cleaning
    vins_before = df.select("vin").distinct().count()
    
    # Filter using info table (removes all rows of outlier VINs)
    df_clean = (
        df
        .join(
            info_vins.filter(f.col("sales_group") == sg),
            on=["vin", "sales_group"],
            how="inner"
        )
    )
    
    # VIN count AFTER cleaning
    vins_after = df_clean.select("vin").distinct().count()
    
    # VIN count in info table for this sales group
    vins_info = (
        info_vins
        .filter(f.col("sales_group") == sg)
        .select("vin")
        .distinct()
        .count()
    )
    
    # Store verification numbers
    verification_rows.append(
        (sg, vins_before, vins_after, vins_info)
    )
    
    # Store cleaned dataframe
    cleaned_spectra_dfs.append(df_clean)

# 5.	SEE IF EVERYTHING MAKES SENSE
# vins_after_cleaning should be equal to vins_info


verification_df = spark.createDataFrame(
    verification_rows,
    ["sales_group", "vins_before_cleaning", "vins_after_cleaning", "vins_in_info"]
)

verification_df.orderBy("sales_group").show(20, truncate=False)

# 6.	COMBINE ALL ENGINE SPECTRAS TOGETHER
from functools import reduce

df_all_engine_spectra_cleaned = reduce(
    lambda d1, d2: d1.unionByName(d2),
    cleaned_spectra_dfs
)

display(df_all_engine_spectra_cleaned)

# 7.	FINAL CHECKS BEFORE WRITING TO CATALOG
# Total VINs across all spectra
df_all_engine_spectra_cleaned.select("vin").distinct().count()

# VINs per sales group
df_all_engine_spectra_cleaned.groupBy("sales_group") \
    .agg(f.countDistinct("vin").alias("vin_count")) \
    .orderBy("sales_group") \
    .show()

# 8.	WRITE DATA TO THE CATALOG
df_all_engine_spectra_cleaned.write.mode("overwrite").saveAsTable(
    "pt_configuration_thesis.alltrucks_v1.engine_spectra_all_cleaned"
)


