package com.databricks.spark.sql.perf.mllib.feature

import scala.util.Random

import org.apache.commons.io.IOUtils

import org.apache.spark.ml
import org.apache.spark.ml.PipelineStage
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql._

import com.databricks.spark.sql.perf.mllib.OptionImplicits._
import com.databricks.spark.sql.perf.mllib.data.SentenceGenerator
import com.databricks.spark.sql.perf.mllib.{BenchmarkAlgorithm, MLBenchContext, TestFromTraining}


object HashingTF extends BenchmarkAlgorithm with TestFromTraining with UnaryTransformer {

  override def trainingDataSet(ctx: MLBenchContext): DataFrame = {
    import ctx.params._
    import ctx.sqlContext.implicits._
    val rng = ctx.newGenerator()
    // Load in a dictionary of ~700 words from a Sherlock Holmes novel, remove non-alphanumeric
    // chars, then construct sentences by randomly sampling from it (to mimic sentences from a
    // real corpus)
    val dictionary = IOUtils.toString(this.getClass.getClassLoader
      .getResourceAsStream(s"sherlockholmes.txt"))
      .replaceAll("[^A-Za-z0-9]", " ").split(' ').filter(_.length > 0)

    val sentenceGenerator = new SentenceGenerator(dictionary, hashingTFSentenceLength)
    RandomRDDs.randomRDD(ctx.sqlContext.sparkContext, sentenceGenerator,
      numExamples, numPartitions, ctx.seed()).toDF(inputCol)
  }

  override def getPipelineStage(ctx: MLBenchContext): PipelineStage = {
    import ctx.params._
    val rng = ctx.newGenerator()
    new ml.feature.HashingTF()
      .setInputCol(inputCol)
      .setNumFeatures(hashingTFNumFeatures)
  }

}
