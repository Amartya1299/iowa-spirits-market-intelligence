# 🥃 Scalable Market Intelligence and Consumer Segmentation in Iowa Spirit Distribution

A scalable Big Data pipeline for analyzing Iowa liquor sales patterns, built on a Docker-containerized Hadoop cluster. The project combines distributed data ingestion, PySpark ETL, Spark SQL analytics, MLlib-based customer segmentation, and interactive Power BI dashboards.

---

## 📌 Project Overview

This end-to-end pipeline processes large-scale Iowa Liquor Sales data to uncover actionable market intelligence — identifying top-performing products, high-revenue counties, seasonal trends, and distinct consumer segments using unsupervised machine learning.

---

## 🏗️ Architecture

```
Raw CSV Data
    │
    ▼
HDFS Ingestion (Hadoop Distributed File System)
    │
    ▼
PySpark ETL (Data Cleaning & Transformation)
    │
    ▼
Spark SQL (8 Analytical Queries)
    │
    ▼
MLlib K-Means Clustering (k=5 Segments)
    │
    ▼
Parquet Output → Power BI Dashboard (5 Pages)
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Distributed Storage | HDFS (Hadoop 3.x) |
| Processing Engine | Apache Spark / PySpark |
| Analytics | Spark SQL |
| Machine Learning | MLlib (K-Means Clustering) |
| Serialization | Apache Parquet |
| Visualization | Microsoft Power BI |
| Infrastructure | Docker, Ubuntu (VirtualBox) |
| Language | Python 3 |

---

## 📂 Project Structure

```
iowa-spirits-market-intelligence/
│
├── final_parquet_convert.py     # Converts processed data to Parquet format
├── spark_sql.py                 # 8 Spark SQL analytical queries
├── kmeans_final.py              # K-Means clustering pipeline (k=5)
├── kmeans_categories.py         # Cluster label assignment and interpretation
├── evaluate_model.py            # Model evaluation and metrics
├── BI Visuals.pdf               # Power BI dashboard export (5 pages)
└── README.md
```

---

## 📊 Key Analyses

### Spark SQL Queries
- Top-selling products and categories by volume and revenue
- County-level and city-level sales distribution
- Seasonal and monthly sales trends
- Vendor performance analysis
- Store-level sales rankings

### K-Means Clustering (k=5)
Consumer segments identified based on purchase frequency, volume, and product category preferences:

| Cluster | Profile |
|---|---|
| 0 | High-volume bulk buyers |
| 1 | Premium spirit purchasers |
| 2 | Budget-conscious buyers |
| 3 | Seasonal/occasional buyers |
| 4 | Diverse category explorers |

---

## 📈 Power BI Dashboard

The 5-page interactive dashboard covers:
1. Sales Overview & KPIs
2. Geographic Distribution (County/City)
3. Product & Category Performance
4. Vendor & Store Analysis
5. Consumer Segment Profiles

📄 Dashboard export: [`BI Visuals.pdf`](./BI%20Visuals.pdf)
> Full `.pbix` file available on request.

---

## 🚀 How to Run

### Prerequisites
- Docker Desktop
- Hadoop 3.x cluster (Docker container)
- Apache Spark 3.x
- Python 3.8+
- PySpark

### Steps

```bash
# 1. Start Hadoop cluster (Docker)
docker-compose up -d

# 2. Upload raw CSV data to HDFS
hdfs dfs -put iowa_liquor_sales.csv /data/

# 3. Run ETL and convert to Parquet
python final_parquet_convert.py

# 4. Run Spark SQL analytics
python spark_sql.py

# 5. Run K-Means clustering
python kmeans_final.py

# 6. Evaluate model
python evaluate_model.py
```

---

## 📁 Dataset

**Iowa Liquor Sales** — Public dataset from the Iowa Department of Commerce  
Source: [Iowa Data Portal](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy)

---

## 👤 Author

**Amartya Mishra**  
MS Computer Science — Purdue University Northwest  
[LinkedIn](https://www.linkedin.com/in/amartya-mishra) | [GitHub](https://github.com/Amartya1299)
