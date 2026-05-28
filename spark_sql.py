from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, round, desc, rank
from pyspark.sql.window import Window
import time
from pyspark.sql.functions import round as spark_round
import builtins
round = builtins.round  # restore Python's built-in round

# ── 1. Start Spark Session ────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Iowa Liquor - Spark SQL Market Analytics") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✓ Spark session started for Phase 3 - Market Analytics")

# ── 2. Load Cleaned Parquet Data from HDFS ───────────────────────────────────
start_total = time.time()

input_path = "hdfs://localhost:9000/user/amartya/data/cleaned_liquor_data"
df = spark.read.parquet(input_path)

print(f"✓ Loaded {df.count():,} rows from cleaned_liquor_data")

# ── 3. Register as Temp View for Spark SQL ───────────────────────────────────
df.createOrReplaceTempView("liquor_sales")
print("✓ Registered liquor_sales as Spark SQL temp view")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 1: Top 10 Counties by Total Revenue
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q1: Top 10 Counties by Revenue ---")
start = time.time()

top_counties = spark.sql("""
    SELECT 
        County,
        ROUND(SUM(`Sale (Dollars)`), 2)     AS total_revenue,
        SUM(`Bottles Sold`)                  AS total_bottles,
        COUNT(*)                             AS transaction_count,
        ROUND(AVG(`Sale (Dollars)`), 2)      AS avg_sale
    FROM liquor_sales
    WHERE County IS NOT NULL AND County != 'UNKNOWN'
    GROUP BY County
    ORDER BY total_revenue DESC
    LIMIT 10
""")

top_counties.show(truncate=False)
print(f"✓ Q1 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 2: Top 10 Vendors by Total Revenue
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q2: Top 10 Vendors by Revenue ---")
start = time.time()

top_vendors = spark.sql("""
    SELECT
        `Vendor Name`,
        ROUND(SUM(`Sale (Dollars)`), 2)     AS vendor_revenue,
        SUM(`Bottles Sold`)                  AS total_bottles,
        COUNT(DISTINCT `Store Number`)       AS stores_supplied,
        ROUND(AVG(`Sale (Dollars)`), 2)      AS avg_sale_per_txn
    FROM liquor_sales
    WHERE `Vendor Name` IS NOT NULL AND `Vendor Name` != 'UNKNOWN'
    GROUP BY `Vendor Name`
    ORDER BY vendor_revenue DESC
    LIMIT 10
""")

top_vendors.show(truncate=False)
print(f"✓ Q2 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 3: Top 10 Spirit Categories by Volume & Revenue
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q3: Top 10 Categories by Volume ---")
start = time.time()

top_categories = spark.sql("""
    SELECT
        `Category Name`,
        SUM(`Bottles Sold`)                  AS total_bottles,
        ROUND(SUM(`Sale (Dollars)`), 2)      AS total_revenue,
        ROUND(AVG(`ProfitPerBottle`), 2)     AS avg_profit_per_bottle,
        ROUND(AVG(`StateMarkupPct`), 2)      AS avg_markup_pct
    FROM liquor_sales
    WHERE `Category Name` IS NOT NULL AND `Category Name` != 'UNKNOWN'
    GROUP BY `Category Name`
    ORDER BY total_bottles DESC
    LIMIT 10
""")

top_categories.show(truncate=False)
print(f"✓ Q3 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 4: Top 10 Stores by Revenue
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q4: Top 10 Stores by Revenue ---")
start = time.time()

top_stores = spark.sql("""
    SELECT
        `Store Name`,
        City,
        County,
        ROUND(SUM(`Sale (Dollars)`), 2)     AS total_revenue,
        SUM(`Bottles Sold`)                  AS total_bottles,
        COUNT(*)                             AS transaction_count
    FROM liquor_sales
    WHERE `Store Name` IS NOT NULL
    GROUP BY `Store Name`, City, County
    ORDER BY total_revenue DESC
    LIMIT 10
""")

top_stores.show(truncate=False)
print(f"✓ Q4 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 5: Quarterly Sales Trend (Temporal Analysis)
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q5: Quarterly Sales Trend ---")
start = time.time()

quarterly_trend = spark.sql("""
    SELECT
        YEAR(Date)                           AS year,
        QUARTER(Date)                        AS quarter,
        ROUND(SUM(`Sale (Dollars)`), 2)      AS quarterly_revenue,
        SUM(`Bottles Sold`)                  AS quarterly_bottles,
        COUNT(*)                             AS transaction_count
    FROM liquor_sales
    WHERE Date IS NOT NULL
    GROUP BY YEAR(Date), QUARTER(Date)
    ORDER BY year ASC, quarter ASC
""")

quarterly_trend.show(truncate=False)
print(f"✓ Q5 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 6: Top 10 Items (SKUs) by Bottles Sold
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q6: Top 10 SKUs by Volume ---")
start = time.time()

top_skus = spark.sql("""
    SELECT
        `Item Description`,
        `Category Name`,
        SUM(`Bottles Sold`)                  AS total_bottles,
        ROUND(SUM(`Sale (Dollars)`), 2)      AS total_revenue,
        ROUND(AVG(`ProfitPerBottle`), 2)     AS avg_profit
    FROM liquor_sales
    WHERE `Item Description` IS NOT NULL
    GROUP BY `Item Description`, `Category Name`
    ORDER BY total_bottles DESC
    LIMIT 10
""")

top_skus.show(truncate=False)
print(f"✓ Q6 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 7: Market Summary Statistics
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q7: Overall Market Summary ---")
start = time.time()

market_summary = spark.sql("""
    SELECT
        COUNT(*)                                    AS total_transactions,
        ROUND(SUM(`Sale (Dollars)`), 2)             AS total_revenue,
        SUM(`Bottles Sold`)                         AS total_bottles_sold,
        ROUND(SUM(`Volume Sold (Liters)`), 2)       AS total_volume_liters,
        ROUND(AVG(`ProfitPerBottle`), 2)            AS avg_profit_per_bottle,
        ROUND(AVG(`StateMarkupPct`), 2)             AS avg_markup_pct,
        COUNT(DISTINCT `Store Number`)              AS unique_stores,
        COUNT(DISTINCT `Vendor Name`)               AS unique_vendors,
        COUNT(DISTINCT `Category Name`)             AS unique_categories,
        COUNT(DISTINCT County)                      AS unique_counties
    FROM liquor_sales
""")

market_summary.show(truncate=False)
print(f"✓ Q7 completed in {round(time.time() - start, 3)}s")

# ═════════════════════════════════════════════════════════════════════════════
# QUERY 8: City-Level Revenue Ranking
# ═════════════════════════════════════════════════════════════════════════════
print("\n--- Q8: Top 10 Cities by Revenue ---")
start = time.time()

top_cities = spark.sql("""
    SELECT
        City,
        County,
        ROUND(SUM(`Sale (Dollars)`), 2)     AS city_revenue,
        SUM(`Bottles Sold`)                  AS city_bottles,
        COUNT(DISTINCT `Store Number`)       AS store_count
    FROM liquor_sales
    WHERE City IS NOT NULL AND City != 'UNKNOWN'
    GROUP BY City, County
    ORDER BY city_revenue DESC
    LIMIT 10
""")

top_cities.show(truncate=False)
print(f"✓ Q8 completed in {round(time.time() - start, 3)}s")

# ── 4. Save Key Outputs to HDFS ──────────────────────────────────────────────
print("\n✓ Saving query results to HDFS...")

output_base = "hdfs://localhost:9000/user/amartya/data/sql_outputs"

top_counties.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/top_counties")
top_vendors.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/top_vendors")
top_categories.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/top_categories")
top_stores.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/top_stores")
quarterly_trend.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/quarterly_trend")
top_skus.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/top_skus")
market_summary.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/market_summary")
top_cities.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{output_base}/top_cities")

total_time = round(time.time() - start_total, 3)
print(f"\n{'='*50}")
print(f"✓ Phase 3 Complete — All 8 SQL queries executed")
print(f"✓ Total execution time: {total_time}s")
print(f"✓ Outputs saved to: {output_base}")
print(f"{'='*50}")

spark.stop()
