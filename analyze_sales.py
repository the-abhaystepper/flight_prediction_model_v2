import os
import sys
import urllib.request

# Suppress PySpark deprecation warnings and set Python paths
os.environ['JAVA_HOME'] = r"C:\Program Files\Java\openlogic-openjdk-17.0.18+8-windows-x64"
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Fix Hadoop Home on Windows to resolve access0 UnsatisfiedLinkError
base_dir = os.path.dirname(os.path.abspath(__file__))
hadoop_home = os.path.join(base_dir, "hadoop")
hadoop_bin = os.path.join(hadoop_home, "bin")

os.makedirs(hadoop_bin, exist_ok=True)

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Download winutils.exe and hadoop.dll
hadoop_version = "hadoop-3.3.0"
base_url = f"https://github.com/cdarlint/winutils/raw/master/{hadoop_version}/bin/"

download_file(base_url + "winutils.exe", os.path.join(hadoop_bin, "winutils.exe"))
download_file(base_url + "hadoop.dll", os.path.join(hadoop_bin, "hadoop.dll"))

# Set environment variables for Hadoop
os.environ['HADOOP_HOME'] = hadoop_home
if hadoop_bin not in os.environ['PATH']:
    os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']

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
