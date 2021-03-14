package lineregression

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.regression.{ LabeledPoint, LinearRegressionWithSGD }
import org.apache.spark.sql.{SparkSession,DataFrame,SQLContext}
import org.apache.spark.sql.Row
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler

/**
  * pengjian 2021/1/29
  * 一元、多元线性拟合
  * 计算运行成功
  */

object App {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("kimiYang");
    val sc = new SparkContext(conf);
    val sqc=new SQLContext(sc)

    val spark=  SparkSession.builder()
      .appName("test")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    import spark.implicits._

    //val data = sc.textFile("/test/kimi.txt");
    // val training = spark.read.format("libsvm").load(data_path)
    //val data = spark.read.format("libsvm").load("E:\\lineregressionsample/test.txt")
    //val data = spark.read.format("libsvm").load("E:\\lineregressionsample/RMHVLibsvm.txt")
    val data = spark.read.format("libsvm").load("E:\\lineregressionsample/multiRMHVLibsvm.txt")

    data.show();
    //val model = LinearRegressionWithSGD.train(parseData, 100, 0.1) //建立模型
    // 建立模型，预测谋杀率Murder
    // 设置线性回归参数
    val lr1 = new LinearRegression()
    val lr2 = lr1.setFitIntercept(true)
    // RegParam：正则化
    val lr3 = lr2.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val lr = lr3

    // 将训练集合代入模型进行训练
    val model = lr.fit(data)
    // 输出模型全部参数
    model.extractParamMap()
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


    //通过模型预测模型
    // 对样本进行测试
    // 模型进行评价
    val trainingSummary = model.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")//RMSE:均方根差
    println(s"r2: ${trainingSummary.r2}")//r2:判定系数，也称为拟合优度，越接近1越好
    trainingSummary.predictions.show()
    sc.stop
  }
}
