import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, StringIndexer, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.rdd.RDD

object spam {

  def main(args: Array[String]): Unit = {
    System.setProperty("spark.executor.memory", "10G")

    val spark = SparkSession
      .builder
      .master("local")
      .appName("Classification")
      .getOrCreate()

    val spam_training = spark.read.textFile("spam_training.txt")
    val spam_testing = spark.read.textFile("spam_testing.txt")
    val nospam_training = spark.read.text("nospam_training.txt")
    val nospam_testing = spark.read.text("nospam_testing.txt")

    val spam_training_DF = spam_training.toDF()
    val spam_testing_DF = spam_testing.toDF()
    val nospam_training_DF = nospam_training.toDF()
    val nospam_testing_DF = nospam_testing.toDF()

    //val spam_training_RDD : RDD[String] = spam_training.rdd
    //val spam_testing_RDD : RDD[String] = spam_testing.rdd

    spam_training.cache()
    spam_testing.cache()
    //spam_training_DF.show(false)

    val indexer = new StringIndexer()
      .setInputCol("value")
      .setOutputCol("label")
      .setHandleInvalid("skip")

    // ML pipeline
    val tokenizer = new Tokenizer()
      .setInputCol("value")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

//    val idf = new IDF()
//      .setInputCol(hashingTF.getOutputCol)
//      .setOutputCol("IDFfeatures")

    val classifier = new NaiveBayes()
      .setSmoothing(1.0)
      .setModelType("multinomial")

    val pipeline = new Pipeline()
      .setStages(Array(indexer, tokenizer, hashingTF, classifier))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val grid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100, 500, 1000))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(grid)
      .setNumFolds(3)

    val model = crossValidator.fit(spam_training_DF)

    val pr = model.transform(spam_testing)
    val metric = evaluator.evaluate(pr)

//    model.write.overwrite().save("outputs/spark-naive-bayes-model")
//    pipeline.write.overwrite().save("outputs/pipeline-model")
//    pr.write.save("outputs/prediction-model")

  }
}