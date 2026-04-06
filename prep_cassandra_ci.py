import os
import sys
import findspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

findspark.init()

#Use local address for CI runner coordination
if sys.platform == 'win32':
    os.environ['JAVA_HOME'] = r"C:\Program Files\OpenJDK\openlogic-openjdk-17.0.18+8-windows-x64"
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

def main():
    print("CI Cassandra starting")
    
    #Initialize session to trigger driver downloads
    spark = SparkSession.builder \
        .appName("CIPrep") \
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
        .config("spark.cassandra.connection.host", "127.0.0.1") \
        .config("spark.sql.catalog.cassandra", "com.datastax.spark.connector.datasource.CassandraCatalog") \
        .getOrCreate()
    
    #Create keyspace on localhost
    print("Initializing Keyspace 'flight_ks'...")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS cassandra.flight_ks WITH PROPERTIES ('replication' = \"{'class':'SimpleStrategy', 'replication_factor':1}\")")
    
    #Check if there is data to load
    if os.path.exists("flight.csv"):
        print("Feeding flight.csv rows into Cassandra for CI training run...")
        df = spark.read.option("header", "true").option("inferSchema", "true").csv("flight.csv")
        
        #Normalize columns for Cassandra
        for col_name in df.columns:
            df = df.withColumnRenamed(col_name, col_name.upper())
        
        #Primary key (year, month, day, airline, origin_airport)
        df.writeTo("cassandra.flight_ks.flights").create()
        print("Ingestion Finished.")
    else:
        print("Warning: flight.csv missing. CI training run might fail later.")
    
    spark.stop()

if __name__ == "__main__":
    main()
