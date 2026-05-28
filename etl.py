from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, round, regexp_replace

# ── 1. Start Spark session with Memory Tuning ─────────────────────────────────
spark = SparkSession.builder \
    .appName("Iowa Liquor ETL - Large Scale") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✓ Spark session started - Memory optimized for 10.4M rows")

# ── 2. Read raw CSV from HDFS (Using the specific path) ──────────────────────
# Note: Filename updated to match the 'ls' output
input_path = "hdfs://localhost:9000/user/amartya/data/Iowa_Liquor_Sales_.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(input_path)

# Useful for large datasets: cache the raw DF if enough RAM is there
# df.cache() 

print(f"✓ Loaded {df.count():,} rows from 2021-2024 range")

# ── 3. Parse WKT store location → latitude & longitude ───────────────────────
df = df.withColumn(
    "longitude",
    regexp_extract(col("Store Location"), r"POINT \((-?\d+\.\d+)", 1).cast("double")
).withColumn(
    "latitude",
    regexp_extract(col("Store Location"), r"POINT \(-?\d+\.\d+ (-?\d+\.\d+)", 1).cast("double")
)

# ── 4. Null imputation ────────────────────────────────────────────────────────
df = df.fillna({
    "County":               "UNKNOWN",
    "Category Name":        "UNKNOWN",
    "Vendor Name":          "UNKNOWN",
    "Bottles Sold":         0,
    "Sale (Dollars)":       0.0,
    "Volume Sold (Liters)": 0.0
})

# ── 5. Clean Currency Symbols and Type Casting ───────────────────────────────
# Cleaning $ signs before casting to double
df = df.withColumn("Bottles Sold", col("Bottles Sold").cast("integer")) \
       .withColumn("State Bottle Cost", regexp_replace(col("State Bottle Cost"), "\\$", "").cast("double")) \
       .withColumn("State Bottle Retail", regexp_replace(col("State Bottle Retail"), "\\$", "").cast("double")) \
       .withColumn("Sale (Dollars)", regexp_replace(col("Sale (Dollars)"), "\\$", "").cast("double")) \
       .withColumn("Volume Sold (Liters)", col("Volume Sold (Liters)").cast("double"))

# ── 6. Feature engineering ────────────────────────────────────────────────────
df = df.withColumn("ProfitPerBottle", round(col("State Bottle Retail") - col("State Bottle Cost"), 2))

df = df.withColumn(
    "StateMarkupPct",
    round(
        when(col("State Bottle Cost") > 0,
            (col("State Bottle Retail") - col("State Bottle Cost")) / col("State Bottle Cost") * 100
        ).otherwise(0.0), 2
    )
)

# ── 7. Preview ────────────────────────────────────────────────────────────────
df.select("Store Name", "County", "Sale (Dollars)", "ProfitPerBottle").show(5)

# ── 8. Write to HDFS as Parquet (Optimized for Storage) ──────────────────────
output_path = "hdfs://localhost:9000/user/amartya/data/cleaned_liquor_data"

# Repartitioning to 10 files makes the K-Means phase MUCH faster later
df.repartition(10).write \
  .mode("overwrite") \
  .parquet(output_path)

print(f"✓ ETL complete. 10.4M rows processed and saved to {output_path}")

spark.stop()
