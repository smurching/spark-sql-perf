package com.databricks.spark.sql.perf.mllib.feature

import scala.util.Random

import org.apache.commons.io.IOUtils

import org.apache.spark.ml
import org.apache.spark.ml.PipelineStage
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql._

import com.databricks.spark.sql.perf.mllib.OptionImplicits._
import com.databricks.spark.sql.perf.mllib.data.DocumentGenerator
import com.databricks.spark.sql.perf.mllib.{BenchmarkAlgorithm, MLBenchContext, TestFromTraining}


object HashingTF extends BenchmarkAlgorithm with TestFromTraining with UnaryTransformer {

  override def trainingDataSet(ctx: MLBenchContext): DataFrame = {
    import ctx.params._
    import ctx.sqlContext.implicits._

    // To test HashingTF, we generate arrays of docLength strings, where
    // each string is selected from a pool of numVocabulary strings
    // The expected # of occurrences of each word in our vocabulary is
    // (docLength * numExamples) / numVocabulary
    val DocumentGenerator = new DocumentGenerator(numVocabulary, docLength)
    RandomRDDs.randomRDD(ctx.sqlContext.sparkContext, DocumentGenerator,
      numExamples, numPartitions, ctx.seed()).toDF(inputCol)
  }

  override def getPipelineStage(ctx: MLBenchContext): PipelineStage = {
    import ctx.params._
    val rng = ctx.newGenerator()
    new ml.feature.HashingTF()
      .setInputCol(inputCol)
      .setNumFeatures(featurizerNumFeatures)
  }

}
