/*
 * Copyright 2015 Databricks Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.databricks.spark.sql.perf

/**
 * The performance results of all given queries for a single iteration.
 *
 * @param timestamp The timestamp indicates when the entire experiment is started.
 * @param iteration The index number of the current iteration.
 * @param tags Tags of this iteration (variations are stored at here).
 * @param configuration Configuration properties of this iteration.
 * @param results The performance results of queries for this iteration.
 */
case class ExperimentRun(
    timestamp: Long,
    iteration: Int,
    tags: Map[String, String],
    configuration: BenchmarkConfiguration,
    results: Seq[BenchmarkResult])

/**
 * The configuration used for an iteration of an experiment.
 *
 * @param sparkVersion The version of Spark.
 * @param sqlConf All configuration properties related to Spark SQL.
 * @param sparkConf All configuration properties of Spark.
 * @param defaultParallelism The default parallelism of the cluster.
 *                           Usually, it is the number of cores of the cluster.
 */
case class BenchmarkConfiguration(
    sparkVersion: String = org.apache.spark.SPARK_VERSION,
    sqlConf: Map[String, String],
    sparkConf: Map[String, String],
    defaultParallelism: Int,
    buildInfo: Map[String, String])

/**
 * The result of a query.
 *
 * @param name The name of the query.
 * @param mode The ExecutionMode of this run.
 * @param parameters Additional parameters that describe this query.
 * @param joinTypes The type of join operations in the query.
 * @param tables The tables involved in the query.
 * @param parsingTime The time used to parse the query.
 * @param analysisTime The time used to analyze the query.
 * @param optimizationTime The time used to optimize the query.
 * @param planningTime The time used to plan the query.
 * @param executionTime The time used to execute the query.
 * @param result the result of this run. It is not necessarily the result of the query.
 *               For example, it can be the number of rows generated by this query or
 *               the sum of hash values of rows generated by this query.
 * @param breakDown The breakdown results of the query plan tree.
 */
case class BenchmarkResult(
    name: String,
    mode: String,
    parameters: Map[String, String] = Map.empty[String, String],
    joinTypes: Seq[String] = Nil,
    tables: Seq[String] = Nil,
    parsingTime: Option[Double] = None,
    analysisTime: Option[Double] = None,
    optimizationTime: Option[Double] = None,
    planningTime: Option[Double] = None,
    executionTime: Option[Double] = None,
    result: Option[Long] = None,
    breakDown: Seq[BreakdownResult] = Nil,
    queryExecution: Option[String] = None,
    failure: Option[Failure] = None,
    mlParams: Option[MLParams] = None,
    mlResult: Option[MLResult] = None)

/**
 * The execution time of a subtree of the query plan tree of a specific query.
 *
 * @param nodeName The name of the top physical operator of the subtree.
 * @param nodeNameWithArgs The name and arguments of the top physical operator of the subtree.
 * @param index The index of the top physical operator of the subtree
 *              in the original query plan tree. The index starts from 0
 *              (0 represents the top physical operator of the original query plan tree).
 * @param executionTime The execution time of the subtree.
 */
case class BreakdownResult(
    nodeName: String,
    nodeNameWithArgs: String,
    index: Int,
    children: Seq[Int],
    executionTime: Double,
    delta: Double)

case class Failure(className: String, message: String)

// KEEP ARGUMENTS SORTED BY NAME.
// It simplifies lookup when checking if a parameter is here already.
case class MLParams(
    // *** Common to all algorithms ***
    randomSeed: Option[Int] = Some(42),
    numExamples: Option[Long] = None,
    numTestExamples: Option[Long] = None,
    numPartitions: Option[Int] = None,
    // *** Specialized and sorted by name ***
    bucketizerNumBuckets: Option[Int] = None,
    depth: Option[Int] = None,
    elasticNetParam: Option[Double] = None,
    family: Option[String] = None,
    hashingTFNumFeatures: Option[Int] = None,
    hashingTFSentenceLength: Option[Int] = None,
    k: Option[Int] = None,
    ldaDocLength: Option[Int] = None,
    ldaNumVocabulary: Option[Int] = None,
    link: Option[String] = None,
    maxIter: Option[Int] = None,
    naiveBayesSmoothing: Option[Double] = None,
    numClasses: Option[Int] = None,
    numFeatures: Option[Int] = None,
    numItems: Option[Int] = None,
    numUsers: Option[Int] = None,
    optimizer: Option[String] = None,
    regParam: Option[Double] = None,
    rank: Option[Int] = None,
    tol: Option[Double] = None
)

object MLParams {
  val empty = MLParams()
}

/**
 * Result information specific to MLlib.
 *
 * @param trainingTime  (MLlib) Training time.
 *                      executionTime is set to the same value to match Spark Core tests.
 * @param trainingMetric  (MLlib) Training metric, such as accuracy
 * @param testTime  (MLlib) Test time (for prediction on test set, or on training set if there
 *                  is no test set).
 * @param testMetric  (MLlib) Test metric, such as accuracy
 */
case class MLResult(
    trainingTime: Option[Double] = None,
    trainingMetric: Option[Double] = None,
    testTime: Option[Double] = None,
    testMetric: Option[Double] = None)
