from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ParquetToCsv").getOrCreate()

# 1. Convert the 10.4M Row ETL Data
print("--- Converting ETL Parquet ---")
spark.read.parquet('/home/vboxuser/final_project_outputs/cleaned_liquor_data').repartition(1).write.csv('etl_master_csv', header=True, mode='overwrite')

# 2. Convert the ML Math (0.486 Result)
print("--- Converting ML Results Parquet ---")
spark.read.parquet('/home/vboxuser/kmeans-final-results').repartition(1).write.csv('ml_math_csv', header=True, mode='overwrite')

spark.stop()
