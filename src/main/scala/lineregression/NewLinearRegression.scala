package lineregression

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

/**
  * pengjian 2021/3/12
  * https://blog.csdn.net/dkl12/article/details/80259410
  * 一元、多元线性拟合
  * 计算运行成功
  */

object NewLinearRegression {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("NewLinearRegression")
      .master("local")
      .getOrCreate()
    //val data_path = "E:\\lineregressionsample/test.txt"
    //抗拉强度RM和硬度HV的拟合数据
   // val data_path = "E:\\lineregressionsample/RMHVLibsvm.txt"//成功
    //val data_path = "E:\\lineregressionsample/RMHVLibsvm2.txt"
    //val data_path = "E:\\lineregressionsample/multiRMHVLibsvm.txt"
    val data_path = "E:\\lineregressionsample/dataLibsvm150A.txt"
    //val data_path = "E:\\lineregressionsample/dataLibsvm150HV.txt"


    import spark.implicits._
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.sql.Row
    val training = spark.read.format("libsvm").load(data_path)

    //iterations迭代次数，就是完成一次epoch所需的batch个数, RegParam：正则化
    val lr = new LinearRegression()
      .setMaxIter(10000)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(training)
    //coefficients 系数 intercept 截距
    println(s"系数                    截距" )
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // 模型进行评价
    val trainingSummary = lrModel.summary
    //numIterations训练迭代次数
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    // residuals样本残差
    trainingSummary.residuals.show()
    //RMSE:均方根差
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    //r2:判定系数，也称为拟合优度，越接近1越好
    println(s"r2: ${trainingSummary.r2}")
    trainingSummary.predictions.show()

    spark.stop()
  }
}
