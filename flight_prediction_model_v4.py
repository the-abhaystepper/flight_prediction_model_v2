import os
import sys
import numpy as np
import findspark
import mlflow

findspark.init()


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LinearSVC, LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Cassandra connection configurations
CASSANDRA_HOST = "127.0.0.1"
KEYSPACE = "flight_ks"
TABLE_READ = "flights"
TABLE_WRITE = "predictions" 

#Set Python paths
os.environ['JAVA_HOME'] = r"C:\Program Files\OpenJDK\openlogic-openjdk-17.0.18+8-windows-x64"

os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


def main():
    print("Initializing Spark Session with Cassandra Connector...")
    spark = SparkSession.builder \
        .appName("FlightDelayPredictionCassandra") \
        .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
        .config("spark.cassandra.connection.host", CASSANDRA_HOST) \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    print(f"\nPulling data from Cassandra ({KEYSPACE}.{TABLE_READ})")
    
    #Read distributed df directly from Cassandra
    spark_df = spark.read \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_READ, keyspace=KEYSPACE) \
        .load()

    #Resolve Cassandra case-insensitivity
    for col_name in spark_df.columns:
        spark_df = spark_df.withColumnRenamed(col_name, col_name.upper())

    #Pre-cleaning
    required_cols = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 
                     'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
                     'SCHEDULED_DEPARTURE', 'SCHEDULED_TIME', 
                     'DISTANCE', 'DEPARTURE_DELAY']
    
    spark_df = spark_df.dropna(subset=required_cols)

    print("\nDistributed feature engineering with Spark SQL")

    cont_hour_formula = ((F.col("SCHEDULED_DEPARTURE").cast("int") / 100).cast("int")) + \
                        ((F.col("SCHEDULED_DEPARTURE").cast("int") % 100) / 60.0)

    spark_df = spark_df.withColumn("DEP_HOUR_CONT", F.coalesce(cont_hour_formula, F.lit(12.0)))
    
    PI = 2 * np.pi

    spark_df = spark_df \
        .withColumn("DEP_HOUR_SIN", F.sin(F.col("DEP_HOUR_CONT") * (PI / 24))) \
        .withColumn("DEP_HOUR_COS", F.cos(F.col("DEP_HOUR_CONT") * (PI / 24))) \
        .withColumn("DELAY", F.when(F.col("DEPARTURE_DELAY") > 0, 1).otherwise(0)) \
        .withColumn("MONTH_SIN", F.sin(F.col("MONTH") * (PI / 12))) \
        .withColumn("MONTH_COS", F.cos(F.col("MONTH") * (PI / 12))) \
        .withColumn("DAY_SIN", F.sin(F.col("DAY") * (PI / 31))) \
        .withColumn("DAY_COS", F.cos(F.col("DAY") * (PI / 31))) \
        .withColumn("DOW_SIN", F.sin(F.col("DAY_OF_WEEK") * (PI / 7))) \
        .withColumn("DOW_COS", F.cos(F.col("DAY_OF_WEEK") * (PI / 7)))

    print("Computing airport delay metrics (Target Encoding)")
    origin_mean = spark_df.groupBy("ORIGIN_AIRPORT").agg(F.mean("DELAY").alias("ORIGIN_DELAY_RATE"))
    dest_mean = spark_df.groupBy("DESTINATION_AIRPORT").agg(F.mean("DELAY").alias("DEST_DELAY_RATE"))

    #Join computed weight keys back to distributed matrix
    spark_df = spark_df.join(origin_mean, on="ORIGIN_AIRPORT", how="left") \
                       .join(dest_mean, on="DESTINATION_AIRPORT", how="left") \
                       .fillna(0.2, subset=["ORIGIN_DELAY_RATE", "DEST_DELAY_RATE"])

    print("Balancing dataset across nodes")
    delay_df = spark_df.filter(F.col("DELAY") == 1)
    no_delay_df = spark_df.filter(F.col("DELAY") == 0)

    delay_count = delay_df.count()
    no_delay_count = no_delay_df.count()

    #Determine fraction ratio for balanced distribution approximation
    fraction = delay_count / no_delay_count if no_delay_count > 0 else 1.0
    fraction = min(fraction, 1.0)

    no_delay_sampled = no_delay_df.sample(withReplacement=False, fraction=fraction, seed=42)
    balanced_df = delay_df.union(no_delay_sampled)

    print("Executing one-hot encoding and indexing in Spark")

    #AIRLINE Categorical conversion
    airline_indexer = StringIndexer(inputCol="AIRLINE", outputCol="AIRLINE_INDEX", handleInvalid="skip")
    balanced_df = airline_indexer.fit(balanced_df).transform(balanced_df)

    airline_encoder = OneHotEncoder(inputCol="AIRLINE_INDEX", outputCol="AIRLINE_VEC")
    balanced_df = airline_encoder.fit(balanced_df).transform(balanced_df)

    #Features to Assembler
    feature_cols = ['SCHEDULED_TIME', 'DISTANCE', 'DEP_HOUR_SIN', 'DEP_HOUR_COS',
                    'MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS', 'DOW_SIN', 'DOW_COS',
                    'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE', 'AIRLINE_VEC']

    #Vector Assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    spark_df = assembler.transform(balanced_df)

    print("Splitting data for Stacking (60% Base, 20% Test, 20% Meta)")
    train_base_df, train_meta_df, test_df = spark_df.randomSplit([0.6, 0.2, 0.2], seed=42)

    scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
    scaler_model = scaler.fit(train_base_df)
    train_base_df = scaler_model.transform(train_base_df).cache()
    train_meta_df = scaler_model.transform(train_meta_df).cache()
    test_df = scaler_model.transform(test_df).cache()

    print("\nTraining RF")
    rf = RandomForestClassifier(labelCol="DELAY", featuresCol="features")
    paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).addGrid(rf.numTrees, [20, 50]).build()
    evaluator_auc = BinaryClassificationEvaluator(labelCol="DELAY", metricName="areaUnderROC")

    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator_auc, numFolds=3)
    cvModel = cv.fit(train_base_df)
    rf_best = cvModel.bestModel

    print("\nTraining GBT")
    gbt = GBTClassifier(labelCol="DELAY", featuresCol="features", maxIter=50)
    gbt_model = gbt.fit(train_base_df)

    print("Training SVM Base")
    svm = LinearSVC(labelCol="DELAY", featuresCol="features", maxIter=100)
    svm_model = svm.fit(train_base_df)

    print("\nImplementing stacking meta-classifier")
    p_rf_meta = rf_best.transform(train_meta_df).withColumnRenamed("prediction", "p_rf").drop("rawPrediction", "probability")
    p_gbt_meta = gbt_model.transform(p_rf_meta).withColumnRenamed("prediction", "p_gbt").drop("rawPrediction", "probability")
    pred_meta = svm_model.transform(p_gbt_meta).withColumnRenamed("prediction", "p_svm").drop("rawPrediction")

    meta_assembler = VectorAssembler(inputCols=["p_rf", "p_gbt", "p_svm"], outputCol="meta_features")
    pred_meta = meta_assembler.transform(pred_meta)

    lr_meta = LogisticRegression(labelCol="DELAY", featuresCol="meta_features")
    meta_model = lr_meta.fit(pred_meta)

    print("\nFinal Evaluation")
    p_rf_test_df = rf_best.transform(test_df).withColumnRenamed("prediction", "p_rf").drop("rawPrediction", "probability")
    p_gbt_test_df = gbt_model.transform(p_rf_test_df).withColumnRenamed("prediction", "p_gbt").drop("rawPrediction", "probability")
    pred_test = svm_model.transform(p_gbt_test_df).withColumnRenamed("prediction", "p_svm").drop("rawPrediction")

    pred_test = meta_assembler.transform(pred_test)
    pred_stacked = meta_model.transform(pred_test).withColumnRenamed("prediction", "p_stacked")

    collected = pred_stacked.select("DELAY", "p_rf", "p_gbt", "p_svm", "p_stacked").collect()
    test_labels = [float(r['DELAY']) for r in collected]
    p_rf_test = [float(r['p_rf']) for r in collected]
    p_gbt_test = [float(r['p_gbt']) for r in collected]
    p_svm_test = [float(r['p_svm']) for r in collected]
    p_stacked_test = [float(r['p_stacked']) for r in collected]

    def get_metrics(p, t):
        acc = sum(1 for i in range(len(t)) if p[i] == t[i]) / len(t)
        tp = sum(1 for i in range(len(t)) if p[i] == 1 and t[i] == 1)
        fp = sum(1 for i in range(len(t)) if p[i] == 1 and t[i] == 0)
        fn = sum(1 for i in range(len(t)) if p[i] == 0 and t[i] == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        return acc, prec, rec, f1

    print(f"{'Model':<25} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 65)

    #MLflow Tracking block
    with mlflow.start_run(run_name="flight_stacked_model"):
        mlflow.log_param("features", len(feature_cols))
        
        for name, pr in [("Random Forest (Tuned)", p_rf_test), 
                         ("GBT (Boosting)", p_gbt_test), 
                         ("SVM", p_svm_test),
                         ("STACKED MODEL", p_stacked_test)]:
            a, prc, r, f = get_metrics(pr, test_labels)
            print(f"{name:<25} | {a:.4f}  | {prc:.4f}  | {r:.4f}  | {f:.4f}")
            
            #Log metrics for the final stacked model
            if name == "STACKED MODEL":
                mlflow.log_metric("accuracy", a)
                mlflow.log_metric("precision", prc)
                mlflow.log_metric("recall", r)
                mlflow.log_metric("f1_score", f)

    print(f"\nSaving stacked predictions to Cassandra ({KEYSPACE}.{TABLE_WRITE})")

    if "FLIGHT_ID" in spark_df.columns:
         save_df = pred_stacked.select("FLIGHT_ID", "p_stacked")
         save_df.write \
             .format("org.apache.spark.sql.cassandra") \
             .options(table=TABLE_WRITE, keyspace=KEYSPACE) \
             .mode("append") \
             .save()
         print("Write Success to Cassandra.")
    else:
         print("Missing 'FLIGHT_ID' identifier.")

    print("\nPipeline completed.")
    spark.stop()

if __name__ == "__main__":
    main()
