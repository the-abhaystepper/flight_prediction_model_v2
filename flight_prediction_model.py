import pandas as pd
import numpy as np
import os
import sys
import findspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LinearSVC, LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Suppress PySpark deprecation warnings and set Python paths
os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', '')
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


findspark.init()


def main():
    print("Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("FlightDelayPredictionStacked") \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    print("\n--- Preprocessing and Feature Engineering ---")
    data_path = "flight.csv"
    if not os.path.exists(data_path):
        print(f"Error: Could not find '{data_path}' in the current directory.")
        return

    print("Loading flight.csv...")
    flights_df = pd.read_csv(data_path, low_memory=False)

    # Cleaning
    flights_agg = flights_df[['MONTH','DAY','DAY_OF_WEEK','AIRLINE',
                              'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                              'SCHEDULED_DEPARTURE','SCHEDULED_TIME',
                              'DISTANCE', 'DEPARTURE_DELAY']].copy()
    flights_agg = flights_agg.dropna(axis=0, how="any")

    # Feature Engineering
    print("Computing Continuous and Cyclical Time Features...")
    def hhmm_to_continuous_hour(val):
        try:
            v = int(float(val))
            h = v // 100
            m = v % 100
            return h + m / 60.0
        except: return 12.0 # Neutral midday

    flights_agg['DEP_HOUR_CONT'] = flights_agg['SCHEDULED_DEPARTURE'].apply(hhmm_to_continuous_hour)
    flights_agg['DEP_HOUR_SIN'] = np.sin(flights_agg['DEP_HOUR_CONT'] * (2. * np.pi / 24))
    flights_agg['DEP_HOUR_COS'] = np.cos(flights_agg['DEP_HOUR_CONT'] * (2. * np.pi / 24))
    flights_agg['DELAY'] = np.where(flights_agg['DEPARTURE_DELAY'] > 0, 1, 0)

    # 1. Cyclical Encoding for Month, Day, and Day of Week
    print("Applying Cyclical Encoding to Month and Day Features...")
    flights_agg['MONTH_SIN'] = np.sin(flights_agg['MONTH'] * (2. * np.pi / 12))
    flights_agg['MONTH_COS'] = np.cos(flights_agg['MONTH'] * (2. * np.pi / 12))
    
    flights_agg['DAY_SIN'] = np.sin(flights_agg['DAY'] * (2. * np.pi / 31))
    flights_agg['DAY_COS'] = np.cos(flights_agg['DAY'] * (2. * np.pi / 31))
    
    flights_agg['DOW_SIN'] = np.sin(flights_agg['DAY_OF_WEEK'] * (2. * np.pi / 7))
    flights_agg['DOW_COS'] = np.cos(flights_agg['DAY_OF_WEEK'] * (2. * np.pi / 7))

    # 2. Target/Mean Encoding for High Cardinality Airports
    print("Computing Airport Delay Metrics (Target Encoding)...")
    # Compute mean delay on the NATURAL distribution to avoid bias after balancing
    mean_delay_origin = flights_agg.groupby('ORIGIN_AIRPORT')['DELAY'].mean().to_dict()
    mean_delay_dest = flights_agg.groupby('DESTINATION_AIRPORT')['DELAY'].mean().to_dict()
    flights_agg['ORIGIN_DELAY_RATE'] = flights_agg['ORIGIN_AIRPORT'].map(mean_delay_origin).fillna(0.2)
    flights_agg['DEST_DELAY_RATE'] = flights_agg['DESTINATION_AIRPORT'].map(mean_delay_dest).fillna(0.2)

    # Balancing
    print("Balancing dataset...")
    np.random.seed(42)
    delay_df = flights_agg[flights_agg.DELAY == 1]
    no_delay_df = flights_agg[flights_agg.DELAY == 0].sample(n=len(delay_df), random_state=42)
    balanced_df = pd.concat([delay_df, no_delay_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Encoding
    cat_columns = ['AIRLINE']
    flight_df_encoded = pd.get_dummies(balanced_df, columns=cat_columns, drop_first=True)
    
    feature_cols = ['SCHEDULED_TIME', 'DISTANCE', 'DEP_HOUR_SIN', 'DEP_HOUR_COS',
                    'MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS', 'DOW_SIN', 'DOW_COS',
                    'ORIGIN_DELAY_RATE', 'DEST_DELAY_RATE'] + \
                   [col for col in flight_df_encoded.columns if col.startswith('AIRLINE_')]
    
    X = flight_df_encoded[feature_cols].values.astype(float)
    y = flight_df_encoded['DELAY'].values.astype(float)

    # 1. Convert Pandas DataFrame to Spark DataFrame
    print("Converting Pandas DataFrame to Spark DataFrame...")
    spark_df = spark.createDataFrame(flight_df_encoded)

    # 2. Vector Assembler for all features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    spark_df = assembler.transform(spark_df)

    # 3. Random Split
    print("Splitting data for Stacking (60% Base, 20% Meta, 20% Test)...")
    train_base_df, train_meta_df, test_df = spark_df.randomSplit([0.6, 0.2, 0.2], seed=42)

    # 4. Scaling Fitted on Train Base
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
    scaler_model = scaler.fit(train_base_df)
    train_base_df = scaler_model.transform(train_base_df).cache()
    train_meta_df = scaler_model.transform(train_meta_df).cache()
    test_df = scaler_model.transform(test_df).cache()

    # 1. Automated Cross-Validation for RandomForest
    print("\n--- 1. Spark Cross-Validation (Hyperparameter Tuning for Random Forest) ---")
    rf = RandomForestClassifier(labelCol="DELAY", featuresCol="features")
    
    paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).addGrid(rf.numTrees, [20, 50]).build()
    evaluator_auc = BinaryClassificationEvaluator(labelCol="DELAY", metricName="areaUnderROC")

    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator_auc, numFolds=3)
    cvModel = cv.fit(train_base_df)
    rf_best = cvModel.bestModel
    print(f"Grid search completed using Spark ML tools.")

    # 2. Gradient Boosted Trees
    print("\n--- 2. Training Gradient Boosted Trees (Boosting Ensemble) ---")
    gbt = GBTClassifier(labelCol="DELAY", featuresCol="features", maxIter=50)
    gbt_model = gbt.fit(train_base_df)

    # 3. Linear SVC
    svm = LinearSVC(labelCol="DELAY", featuresCol="features", maxIter=100)
    svm_model = svm.fit(train_base_df)

    # 4. Stacking (Meta-Learning)
    print("\n--- 4. Implementing Stacking (Meta-Classifier) ---")
    # Generating Meta-Features
    print("Generating Meta-Features for Meta-Classifier...")
    p_rf_meta = rf_best.transform(train_meta_df).withColumnRenamed("prediction", "p_rf").drop("rawPrediction", "probability")
    p_gbt_meta = gbt_model.transform(p_rf_meta).withColumnRenamed("prediction", "p_gbt").drop("rawPrediction", "probability")
    pred_meta = svm_model.transform(p_gbt_meta).withColumnRenamed("prediction", "p_svm").drop("rawPrediction")

    meta_assembler = VectorAssembler(inputCols=["p_rf", "p_gbt", "p_svm"], outputCol="meta_features")
    pred_meta = meta_assembler.transform(pred_meta)

    lr_meta = LogisticRegression(labelCol="DELAY", featuresCol="meta_features")
    meta_model = lr_meta.fit(pred_meta)

    # EVALUATION
    print("\n--- Final Evaluation on Test Set ---")
    p_rf_test_df = rf_best.transform(test_df).withColumnRenamed("prediction", "p_rf").drop("rawPrediction", "probability")
    p_gbt_test_df = gbt_model.transform(p_rf_test_df).withColumnRenamed("prediction", "p_gbt").drop("rawPrediction", "probability")
    pred_test = svm_model.transform(p_gbt_test_df).withColumnRenamed("prediction", "p_svm").drop("rawPrediction")

    pred_test = meta_assembler.transform(pred_test)
    pred_stacked = meta_model.transform(pred_test).withColumnRenamed("prediction", "p_stacked")

    # Metrics computation logic
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
    for name, pr in [("Random Forest (Tuned)", p_rf_test), 
                     ("GBT (Boosting)", p_gbt_test), 
                     ("SVM", p_svm_test),
                     ("STACKED MODEL", p_stacked_test)]:
        a, prc, r, f = get_metrics(pr, test_labels)
        print(f"{name:<25} | {a:.4f}  | {prc:.4f}  | {r:.4f}  | {f:.4f}")

    print("\nScript completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
