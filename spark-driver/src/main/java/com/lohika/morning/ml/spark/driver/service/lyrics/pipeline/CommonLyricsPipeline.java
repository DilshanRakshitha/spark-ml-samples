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
import org.apache.spark.ml.param.ParamMap; // Import added
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
// import org.apache.spark.sql.api.java.UDF1; // No longer needed with registration
// import org.apache.spark.sql.expressions.UserDefinedFunction; // No longer needed with registration
import org.apache.spark.sql.types.DataTypes;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

public abstract class CommonLyricsPipeline implements LyricsPipeline {

    @Autowired
    protected SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    @Value("${lyrics.training.set.csv.path}")
    private String lyricsTrainingSetCsvPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    protected int numClasses; // To store the number of actual classes used for training

    @Override
    public GenrePrediction predict(final String unknownLyrics) {
        String lyrics[] = unknownLyrics.split("\\r?\\n");
        Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics),
           Encoders.STRING());

        // Prepare the input DataFrame for prediction
        Dataset<Row> unknownLyricsDataset = lyricsDataset
                .withColumnRenamed("value", VALUE.getName()) // Match training input column
                .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                .withColumn(ID.getName(), monotonically_increasing_id().cast(DataTypes.StringType)); // Add ID if needed


        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());
        System.out.println("Loaded model for prediction. Displaying its training statistics:");
        getModelStatistics(model); // Display stats of the loaded model

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);

        // Handle case where the input might result in no rows after transformations
        // *** Fix: Changed isEmpty() to count() == 0 ***
        if (predictionsDataset.count() == 0) {
            System.out.println("Warning: Prediction resulted in an empty dataset for input: " + unknownLyrics);
            return new GenrePrediction(Genre.UNKNOWN.getName());
        }

        Row predictionRow = predictionsDataset.first();

        System.out.println("\n------------------- PREDICTION -------------------");
        final Double prediction = predictionRow.getAs("prediction");
        System.out.println("Predicted Label: " + prediction);
        Genre predictedGenre = getGenreFromPredictionValue(prediction); // Use the updated method name
        System.out.println("Predicted Genre: " + predictedGenre.getName());


        if (Arrays.asList(predictionsDataset.columns()).contains("probability")) {
            final DenseVector probability = predictionRow.getAs("probability");
            System.out.println("Probability Vector: " + probability);
        }
        System.out.println("------------------------------------------------\n");

        // Return prediction based on the predicted label
        return new GenrePrediction(predictedGenre.getName());
    }

    Dataset<Row> readLyrics() {
        System.out.println("Reading training data from CSV: " + lyricsTrainingSetCsvPath);

        Dataset<Row> rawData = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("escape", "\"")
                .csv(lyricsTrainingSetCsvPath);

        System.out.println("Raw CSV Schema:");
        rawData.printSchema();

        // --- Genre to Label Mapping using Registered UDF ---
        // *** Fix: Register UDF and use expr() ***
        sparkSession.udf().register("genreToLabel", (String genreString) -> {
            if (genreString == null) return Genre.UNKNOWN.getValue();
            return Genre.fromString(genreString.toLowerCase().trim()).getValue(); // Use trim()
        }, DataTypes.DoubleType);


        Dataset<Row> labeledData = rawData
                .withColumnRenamed("lyrics", VALUE.getName()) // Rename lyrics column to 'value'
                 // Apply UDF using Spark SQL expression syntax
                .withColumn(LABEL.getName(), expr("genreToLabel(lower(trim(genre)))"))
                .withColumn(ID.getName(), monotonically_increasing_id().cast(DataTypes.StringType)); // Add unique ID

        // --- Filtering Data ---
        Dataset<Row> preparedData = labeledData
                .filter(col(LABEL.getName()).notEqual(Genre.UNKNOWN.getValue())) // Keep only known genres
                .filter(col(VALUE.getName()).isNotNull())
                .filter(col(VALUE.getName()).notEqual(""))
                .filter(col(VALUE.getName()).contains(" ")) // Keep original filter
                .select(ID.getName(), LABEL.getName(), VALUE.getName()); // Select final columns


        // --- Determine Number of Classes ---
        List<Row> distinctLabels = preparedData.select(LABEL.getName()).distinct().collectAsList();
        this.numClasses = distinctLabels.size();
        System.out.println("Number of distinct classes found in data: " + this.numClasses);
        System.out.println("Distinct labels found: " + distinctLabels.stream()
                .map(r -> r.getDouble(0))
                .sorted()
                .collect(Collectors.toList()));
        System.out.println("Mapping:");
        distinctLabels.stream()
                      .map(r -> r.getDouble(0))
                      .sorted()
                      .forEach(labelVal -> System.out.println("  Label " + labelVal + " -> Genre: " + Genre.fromValue(labelVal).getName()));

        // --- Caching and Validation ---
        preparedData = preparedData.repartition(sparkSession.sparkContext().defaultParallelism()).cache();
        long count = preparedData.count();
        System.out.println("Final count of prepared sentences for training: " + count);


        if (count == 0) {
            throw new RuntimeException("No data available for training after filtering. Check CSV path ('" + lyricsTrainingSetCsvPath + "'), 'genre' column content, and filtering logic.");
        }
        if (this.numClasses < 2) {
            throw new RuntimeException("Not enough classes for classification. Found only " + this.numClasses + " valid class(es) after filtering. Need at least 2.");
        }

        System.out.println("Sample of prepared data for training:");
        preparedData.show(10, false);

        return preparedData;
    }

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

    // Renamed method to avoid conflict with Genre enum
    private Genre getGenreFromPredictionValue(Double value) {
        return Genre.fromValue(value); // Use the static method from Genre enum
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