import os
import sys
import findspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

findspark.init()

# Ensure pytest bypasses windows issues
os.environ['JAVA_HOME'] = r"C:\Program Files\OpenJDK\openlogic-openjdk-17.0.18+8-windows-x64"
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def test_continuous_hour():
    spark = SparkSession.builder \
        .appName("TestPipeline") \
        .master("local[1]") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("ERROR")

    # Mock row using Spark SQL to bypass Spark Pickle/Serialization closures in Pytest
    df = spark.sql("SELECT 1430 AS SCHEDULED_DEPARTURE")
    
    # Run continuous hour algebra (Fixed with correct integer division cast)
    cont_hour_formula = ((F.col("SCHEDULED_DEPARTURE").cast("int") / 100).cast("int")) + \
                        ((F.col("SCHEDULED_DEPARTURE").cast("int") % 100) / 60.0)

    res = df.withColumn("DEP_HOUR_CONT", cont_hour_formula)
    
    # Assert that 1430 equates to 14.5 continuous hours exactly
    evaluated_val = res.collect()[0]["DEP_HOUR_CONT"]
    
    assert evaluated_val == 14.5, f"Expected 14.5, Got {evaluated_val}"
    
    spark.stop()
