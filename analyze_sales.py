import os
import sys

# Suppress PySpark deprecation warnings and set Python paths
os.environ['JAVA_HOME'] = r"C:\Program Files\Java\openlogic-openjdk-17.0.18+8-windows-x64"
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum

def main():
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("SalesDataAnalysis") \
        .getOrCreate()
        
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    input_path = "archive/sales_data_sample.csv"
    output_path = "total_sales_by_product"

    if not os.path.exists(input_path):
        print(f"Error: Could not find '{input_path}'...")
        return

    print("Reading CSV data...")
    # Read the CSV file containing sales data
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Action: Count Original rows
    print(f"Original Row Count: {df.count()}")

    print("Performing Data Cleaning...")
    # Transformation: Handling missing values in required fields (PRODUCTCODE, SALES)
    # We drop any rows where either of these are null
    cleaned_df = df.dropna(subset=['PRODUCTCODE', 'SALES'])
    
    # Transformation: Removing duplicate rows
    cleaned_df = cleaned_df.dropDuplicates()

    # Action: Count Cleaned rows
    print(f"Cleaned Row Count: {cleaned_df.count()}")

    print("Calculating Total Sales by Product...")
    # Transformation: Calculate total sales amount for each product
    product_sales_df = cleaned_df.groupBy("PRODUCTCODE") \
                                 .agg(_sum("SALES").alias("TOTAL_SALES")) \
                                 .orderBy(col("TOTAL_SALES").desc())

    # Action: Display the top 10 results to console
    print("Top 10 Products by Sales Amount:")
    product_sales_df.show(10)

    print(f"Writing Output to '{output_path}' directory (Action)...")
    # Action/Transformation: Output the results to a new CSV file
    product_sales_df.write.csv(output_path, header=True, mode="overwrite")
    
    print("Analysis Process Complete.")

    spark.stop()

if __name__ == "__main__":
    main()
