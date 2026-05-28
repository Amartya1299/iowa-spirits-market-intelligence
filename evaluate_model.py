from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("ModelEvaluation").getOrCreate()

# Load the results
predictions = spark.read.parquet("hdfs://localhost:9000/user/amartya/data/clustering_results")

# Show the columns to debug if it fails
print("Found columns:", predictions.columns)

# 2. Re-create the 'features' vector that the evaluator needs
# We use the same columns used during the training phase
assembler = VectorAssembler(
    inputCols=['latitude', 'longitude', 'Sale (Dollars)'], 
    outputCol="features"
)
data_with_features = assembler.transform(predictions)

# 3. Setup the evaluator 
# Point 'predictionCol' to 'cluster', which is the column containing the cluster IDs
evaluator = ClusteringEvaluator(
    predictionCol="cluster", 
    featuresCol="features", 
    metricName="silhouette", 
    distanceMeasure="squaredEuclidean"
)
#
# Explicitly set the prediction and features columns
# In PySpark ML, 'prediction' is the cluster ID and 'features' is the vector

# 4. Calculate and Print
silhouette = evaluator.evaluate(data_with_features)

print("\n" + "="*40)
print(f"SUCCESS! YOUR SILHOUETTE SCORE IS: {silhouette}")
print("="*40 + "\n")

spark.stop()

