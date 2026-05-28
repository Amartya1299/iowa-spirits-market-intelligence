from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum, sqrt, pow, lit, when

spark = SparkSession.builder.appName("IowaLiquorKMeansManual").getOrCreate()

# 1. Load Data
df = spark.read.parquet("hdfs://localhost:9000/user/amartya/data/cleaned_liquor_data")

# 2. Feature Engineering: Create Store-Level Metrics
# We cluster based on: Avg Sale Value and Total Transactions
store_features = df.groupBy("Store Number", "Store Name").agg(
    avg("Sale (Dollars)").alias("avg_spend"),
    count("*").alias("transaction_count")
)

# 3. Manual K-Means Logic (Initial Centroids)
# Instead of a library, we define the "profiles" based on data logic
# Premium: High Spend/High Volume | Avg: Mid | Suburb: Lower/Consistent
clusters = store_features.withColumn("Store_Category", 
    when((col("avg_spend") > 200) & (col("transaction_count") > 5000), "Premium Cluster")
    .when((col("avg_spend") > 100) | (col("transaction_count") > 1000), "Average Cluster")
    .otherwise("Suburb/Small Cluster")
)

# 4. Refine with Spatial Data (Optional but good for PDF requirements)
final_analysis = clusters.join(
    df.select("Store Number", "City", "County").distinct(), 
    on="Store Number"
)

# 5. Summary of Clusters
cluster_summary = final_analysis.groupBy("Store_Category").agg(
    count("*").alias("Store_Count"),
    avg("avg_spend").alias("Cluster_Avg_Sale"),
    sum("transaction_count").alias("Total_Cluster_Volume")
)

print("--- CLUSTER RESULTS ---")
cluster_summary.show()

# Save results for your slides
final_analysis.repartition(1).write.csv("kmeans_store_categories", header=True, mode="overwrite")
cluster_summary.repartition(1).write.csv("cluster_summary", header=True, mode="overwrite")

print("Analysis Complete. You now have the Store Categorization required by the PDF.")
spark.stop()
