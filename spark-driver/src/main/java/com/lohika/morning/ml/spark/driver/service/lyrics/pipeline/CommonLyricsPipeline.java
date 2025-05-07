package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import static org.apache.spark.sql.functions.*;

import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger; // Import Logger
import org.slf4j.LoggerFactory; // Import LoggerFactory
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

// Mark as abstract since it doesn't implement all methods directly (like getModelDirectory)
public abstract class CommonLyricsPipeline implements LyricsPipeline {

    // *** Add Logger Definition ***
    private static final Logger log = LoggerFactory.getLogger(CommonLyricsPipeline.class);

    @Autowired
    protected SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    @Value("${lyrics.training.set.csv.path}")
    private String lyricsTrainingSetCsvPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    protected int numClasses;

    // *** Removed @Override *** since it's no longer in the interface
    // This method is now effectively unused as prediction logic moved to the Service
    // You could remove it entirely if desired, but keeping it doesn't hurt for now.
    public GenrePrediction predict(final String unknownLyrics) {
         log.warn("CommonLyricsPipeline.predict(String) called directly - This logic has moved to LyricsService. Returning UNKNOWN.");
         // Simplified logic since this shouldn't be the main entry point for prediction anymore
         String lyrics[] = unknownLyrics.split("\\r?\\n");
         Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics), Encoders.STRING());
         Dataset<Row> unknownLyricsDataset = lyricsDataset
                 .withColumnRenamed("value", VALUE.getName())
                 .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                 .withColumn(ID.getName(), functions.monotonically_increasing_id().cast(DataTypes.StringType));

        try {
             CrossValidatorModel model = loadModel();
             PipelineModel bestModel = (PipelineModel) model.bestModel();
             Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);
             if (predictionsDataset.count() == 0) {
                 return new GenrePrediction(Genre.UNKNOWN.getName());
             }
             Row predictionRow = predictionsDataset.first();
             Double predictionLabel = predictionRow.getAs("prediction");
             return new GenrePrediction(getGenreFromPredictionValue(predictionLabel).getName());
        } catch (Exception e) {
             log.error("Error in fallback predict method: {}", e.getMessage());
             return new GenrePrediction(Genre.UNKNOWN.getName());
        }
    }

    // --- readLyrics method remains the same as previous correct version ---
    Dataset<Row> readLyrics() {
        log.info("Reading training data from CSV: {}", lyricsTrainingSetCsvPath); // Use logger

        Dataset<Row> rawData = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("escape", "\"")
                .csv(lyricsTrainingSetCsvPath);

        log.debug("Raw CSV Schema:"); // Use debug for schema
        rawData.printSchema();

        sparkSession.udf().register("genreToLabel", (String genreString) -> {
            if (genreString == null) return Genre.UNKNOWN.getValue();
            return Genre.fromString(genreString.toLowerCase().trim()).getValue();
        }, DataTypes.DoubleType);

        Dataset<Row> labeledData = rawData
                .withColumnRenamed("lyrics", VALUE.getName())
                .withColumn(LABEL.getName(), expr("genreToLabel(lower(trim(genre)))"))
                .withColumn(ID.getName(), monotonically_increasing_id().cast(DataTypes.StringType));

        Dataset<Row> preparedData = labeledData
                .filter(col(LABEL.getName()).notEqual(Genre.UNKNOWN.getValue()))
                .filter(col(VALUE.getName()).isNotNull())
                .filter(col(VALUE.getName()).notEqual(""))
                .filter(col(VALUE.getName()).contains(" "))
                .select(ID.getName(), LABEL.getName(), VALUE.getName());

        List<Row> distinctLabels = preparedData.select(LABEL.getName()).distinct().collectAsList();
        this.numClasses = distinctLabels.size();
        log.info("Number of distinct classes found in data: {}", this.numClasses);
        log.info("Distinct labels found: {}", distinctLabels.stream()
                .map(r -> r.getDouble(0))
                .sorted()
                .collect(Collectors.toList()));
        log.info("Mapping:");
        distinctLabels.stream()
                      .map(r -> r.getDouble(0))
                      .sorted()
                      .forEach(labelVal -> log.info("  Label {} -> Genre: {}", labelVal, Genre.fromValue(labelVal).getName()));

        preparedData = preparedData.repartition(sparkSession.sparkContext().defaultParallelism()).cache();
        long count = preparedData.count();
        log.info("Final count of prepared sentences for training: {}", count);


        if (count == 0) {
            throw new RuntimeException("No data available for training after filtering. Check CSV path ('" + lyricsTrainingSetCsvPath + "'), 'genre' column content, and filtering logic.");
        }
        if (this.numClasses < 2) {
            throw new RuntimeException("Not enough classes for classification. Found only " + this.numClasses + " valid class(es) after filtering. Need at least 2.");
        }

        log.info("Sample of prepared data for training:");
        preparedData.show(10, false);

        return preparedData;
    }
    // --- End readLyrics ---


    // --- loadModel method implementation (using logger) ---
    @Override
    public CrossValidatorModel loadModel() {
        String modelPath = getModelDirectory();
        log.info("Loading model from path: {}", modelPath); // Use logger
        try {
            return mlService.loadCrossValidationModel(modelPath);
        } catch (Exception e) {
            log.error("Failed to load model from {}. Has it been trained?", modelPath, e); // Use logger
            throw new RuntimeException("Model not found or failed to load from: " + modelPath, e);
        }
    }
    // --- End loadModel ---


    // --- Other methods remain the same ---
    protected MulticlassClassificationEvaluator getAccuracyEvaluator() {
        return new MulticlassClassificationEvaluator()
                .setLabelCol(LABEL.getName())
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();
        double[] avgMetrics = model.avgMetrics();

        if (avgMetrics != null && avgMetrics.length > 0) {
            double bestAvgMetric = -1.0;
            int bestParamMapIndex = -1;
            for(int i=0; i<avgMetrics.length; i++){
                if(avgMetrics[i] > bestAvgMetric){
                    bestAvgMetric = avgMetrics[i];
                    bestParamMapIndex = i;
                }
            }
            modelStatistics.put("Best Cross-Validation Accuracy", String.format("%.4f", bestAvgMetric));

            if (bestParamMapIndex != -1) {
                 ParamMap[] bestEstimatorParamMaps = model.getEstimatorParamMaps();
                 modelStatistics.put("Best Parameters", bestEstimatorParamMaps[bestParamMapIndex].toString());
            } else {
                 modelStatistics.put("Best Parameters", "N/A (Index not found)");
            }

        } else {
            modelStatistics.put("Best Cross-Validation Accuracy", "N/A (Metrics unavailable)");
            modelStatistics.put("Best Parameters", "N/A");
        }

        printModelStatistics(modelStatistics);
        return modelStatistics;
    }

    private Genre getGenreFromPredictionValue(Double value) {
        return Genre.fromValue(value);
    }

    void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------- MODEL STATISTICS -------------------");
        modelStatistics.forEach((key, value) -> System.out.println(key + ": " + value));
        System.out.println("------------------------------------------------------\n");
    }

     void saveModel(CrossValidatorModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    void saveModel(PipelineModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    public void setLyricsTrainingSetCsvPath(String lyricsTrainingSetCsvPath) {
        this.lyricsTrainingSetCsvPath = lyricsTrainingSetCsvPath;
    }

    public void setLyricsModelDirectoryPath(String lyricsModelDirectoryPath) {
        this.lyricsModelDirectoryPath = lyricsModelDirectoryPath;
    }

    protected abstract String getModelDirectory();

    String getLyricsModelDirectoryPath() {
        return lyricsModelDirectoryPath;
    }
}