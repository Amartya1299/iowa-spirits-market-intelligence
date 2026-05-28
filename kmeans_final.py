from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize Spark with extra memory settings
spark = SparkSession.builder \
    .appName("FinalScaledKMeans") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# 1. Load Cleaned Data
raw_data = spark.read.parquet("hdfs://localhost:9000/user/amartya/data/cleaned_liquor_data")

# 2. Clean and Sample (Use 10% to prevent Java Gateway crashes)
data = raw_data.dropna(subset=['latitude', 'longitude', 'Sale (Dollars)'])
sampled_data = data.sample(False, 0.1, seed=42) 

# 3. Assemble Features
assembler = VectorAssembler(
    inputCols=['latitude', 'longitude', 'Sale (Dollars)'], 
    outputCol="unscaled_features",
    handleInvalid="skip"
)
assembled_data = assembler.transform(sampled_data)

# 4. Scale Features (Crucial for s >= 0.5)
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

# 5. Run K-Means (k=5)
kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=5, seed=42)
model = kmeans.fit(scaled_data)
predictions = model.transform(scaled_data)

# 6. Evaluating Silhouette Score
evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features")
silhouette = evaluator.evaluate(predictions)

print("\n" + "="*40)
print(f"FINAL SCALED SILHOUETTE SCORE: {silhouette}")
print("="*40 + "\n")

# Calculating WCSS (Within-Cluster Sum of Squares)
wcss = model.summary.trainingCost
print(f"WCSS for k={kmeans.getK()}: {wcss}")

# 7. Save the sample results for visualization
predictions.select('latitude', 'longitude', 'Sale (Dollars)', 'prediction') \
    .write.mode("overwrite") \
    .parquet("hdfs://localhost:9000/user/amartya/data/final_ml_results")

spark.stop()
