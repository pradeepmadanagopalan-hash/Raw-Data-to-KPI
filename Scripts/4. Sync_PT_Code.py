'''
0. README
Notebook: Sales_code_addition
Inputs
•	pt_configuration_thesis.alltrucks_v1.vin_info_sg_info_cluster_info_talpy_info_mds_info_cf_info
•	customer_data_analytics.truck_live.vehicle_info_enriched
Outputs
•	pt_configuration_thesis.alltrucks_v1.sales_group_mapping
•	pt_configuration_thesis.alltrucks_v1.vin_info_fin_info_sgmap_info_cluster_info_talpy_info_mds_info_cf_info
Purpose
•	This notebook is meant to add 6 new columns (all metadata) to the last updates global info table vin_info_sg_info_cluster_info_talpy_info_mds_info_cf_info.
•	The 6 new columns added are vehicle_model, engine_description, transmission_description, rear_axle_ratio_description, tire_size_description and fin.
•	Additionally a dataframe called sales_group_mapping is created and written to the catalog. This table maps the sales group number (1 - 15) to the PT config codes also as a stand alone table.
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
info = spark.table("pt_configuration_thesis.alltrucks_v1.`vin_info_sg_info_cluster_info_talpy_info_mds_info_cf_info`")

unique_vins_count_ori = info.select("vin").distinct().count()
print(f"Number of vehicles in the dataset: {unique_vins_count_ori}")

df_truck_live_vehicle_info_enriched = spark.table("customer_data_analytics.truck_live.vehicle_info_enriched")

# --- Select only the needed columns from second table, including fin ---
df_config = df_truck_live_vehicle_info_enriched.select(
    "vin",
    f.col("vehicle_model").alias("vehicle_model"),
    f.col("code_group_powertrain_engine_power_rating_code_description_en").alias("engine_description"),
    f.col("code_group_powertrain_transmission_code_description_en").alias("transmission_description"),
    f.col("code_group_powertrain_axle_ratio_code_description_en").alias("rear_axle_ratio_description"),
    f.col("code_group_wheels_rims_ra_code_description_en").alias("tire_size_description"),
    "fin"  
)

# --- Join to add the 6 new columns ---
info_updated = (
    info
    .join(df_config, on="vin", how="left")
)

display(info_updated)

# Cross Checking to see if everything makes sense :)

distinct_counts = {
    "engine_description": info_updated.select("engine_description").distinct().count(),
    "transmission_description": info_updated.select("transmission_description").distinct().count(),
    "rear_axle_ratio_description": info_updated.select("rear_axle_ratio_description").distinct().count(),
    "tire_size_description": info_updated.select("tire_size_description").distinct().count(),
    "vehicle_model": info_updated.select("vehicle_model").distinct().count()
}

for col, count in distinct_counts.items():
    print(f"Distinct {col}: {count}")


# Again cross checking to see if everything makes sense :))

config_cols = [
    "vehicle_model",
    "engine_description",
    "transmission_description",
    "rear_axle_ratio_description",
    "tire_size_description"
]

vin_counts_by_config = (
    info_updated
        .groupBy(config_cols)
        .agg(f.countDistinct("vin").alias("vin_count"))
        .orderBy(f.desc("vin_count"))
)

display(vin_counts_by_config)

# --- Select only the required columns ---
sales_group_mapping = info_updated.select(
    "vin",
    "sales_group",
    "vehicle_model",
    "engine_description",
    "transmission_description",
    "rear_axle_ratio_description",
    "tire_size_description"
)

display(sales_group_mapping)

# 3.	PUSH DATA TO THE CATALOG
sales_group_mapping.write.mode("overwrite").saveAsTable(
    "pt_configuration_thesis.alltrucks_v1.sales_group_mapping"
)

info_updated.write.mode("overwrite").saveAsTable(
    "pt_configuration_thesis.alltrucks_v1.vin_info_fin_info_sgmap_info_cluster_info_talpy_info_mds_info_cf_info"
)

