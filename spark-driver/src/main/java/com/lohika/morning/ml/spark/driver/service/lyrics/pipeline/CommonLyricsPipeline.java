package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import static org.apache.spark.sql.functions.*; // Add this import

import com.lohika.morning.ml.spark.driver.service.MLService;
import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes; // Add this import
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

public abstract class CommonLyricsPipeline implements LyricsPipeline {

    @Autowired
    protected SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    // Change this property name to match application.properties
    @Value("${lyrics.training.set.csv.path}")
    private String lyricsTrainingSetCsvPath;

    // Keep this for the output model directory
    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Override
    public GenrePrediction predict(final String unknownLyrics) {
        String lyrics[] = unknownLyrics.split("\\r?\\n");
        Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics),
           Encoders.STRING());

        // Prepare the input DataFrame for prediction - needs 'value' column
        Dataset<Row> unknownLyricsDataset = lyricsDataset
                // Rename 'value' column from createDataset to match the 'value' column used in training read
                .withColumnRenamed("value", VALUE.getName())
                .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                // Add a dummy ID if needed by any transformer, otherwise might not be necessary for prediction
                .withColumn(ID.getName(), monotonically_increasing_id().cast(DataTypes.StringType)); // Add ID for prediction


        CrossValidatorModel model = mlService.loadCrossValidationModel(getModelDirectory());
        getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();

        Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);
        Row predictionRow = predictionsDataset.first();

        System.out.println("\n------------------------------------------------");
        final Double prediction = predictionRow.getAs("prediction");
        System.out.println("Prediction: " + Double.toString(prediction));

        if (Arrays.asList(predictionsDataset.columns()).contains("probability")) {
            final DenseVector probability = predictionRow.getAs("probability");
            System.out.println("Probability: " + probability);
            System.out.println("------------------------------------------------\n");

            return new GenrePrediction(getGenre(prediction).getName(), probability.apply(0), probability.apply(1));
        }

        System.out.println("------------------------------------------------\n");
        return new GenrePrediction(getGenre(prediction).getName());
    }

    // --- REFACTORED DATA READING ---
    Dataset<Row> readLyrics() {
        System.out.println("Reading training data from CSV: " + lyricsTrainingSetCsvPath);

        // Read the CSV file
        Dataset<Row> rawData = sparkSession.read()
                .option("header", "true") // Use first line as header
                .option("inferSchema", "true") // Infer column types (might need manual schema for large datasets)
                .option("escape", "\"") // Handle quotes within lyrics if any
                .csv(lyricsTrainingSetCsvPath);

        // Filter for only 'metal' and 'pop' genres (as the original code focused on binary classification)
        // Adjust this if you want multi-class classification later
        Dataset<Row> filteredData = rawData
                .filter(col("genre").equalTo("metal").or(col("genre").equalTo("pop")));

        System.out.println("Total sentences after filtering for metal/pop: " + filteredData.count());


        // Prepare the DataFrame for the pipeline
        Dataset<Row> preparedData = filteredData
                // Select the lyrics column and rename it to 'value' as expected by Cleanser
                .withColumnRenamed("lyrics", VALUE.getName())
                // Create the numeric 'label' column based on the 'genre' text column
                // Metal = 0.0, Pop = 1.0 (matching the original Genre enum)
                .withColumn(LABEL.getName(),
                        when(col("genre").equalTo("metal"), lit(0.0))
                        .otherwise(lit(1.0)) // Assuming anything not metal is pop after filtering
                )
                 // Add a unique ID column - needed for downstream transformers like Numerator/Exploder/Uniter/Verser
                .withColumn(ID.getName(), monotonically_increasing_id().cast(DataTypes.StringType))
                // Select only the columns needed for the pipeline start
                .select(ID.getName(), LABEL.getName(), VALUE.getName());

        // Filter out potential nulls or empty strings in the 'value' column after processing
        preparedData = preparedData
                            .filter(col(VALUE.getName()).isNotNull())
                            .filter(col(VALUE.getName()).notEqual(""))
                            .filter(col(VALUE.getName()).contains(" ")); // Keep original filter


        // Reduce partitions and cache
        preparedData = preparedData.coalesce(sparkSession.sparkContext().defaultMinPartitions()).cache();
        // Force caching.
        long count = preparedData.count();
        System.out.println("Final count of prepared sentences: " + count);

        // Show a sample to verify
        System.out.println("Sample of prepared data:");
        preparedData.show(5, false); // Show 5 rows, don't truncate

        return preparedData;
    }

    // This method is no longer needed as we read from a single CSV
    /*
    private Dataset<Row> readLyricsForGenre(String inputDirectory, Genre genre) {
        // ... original code ...
    }
    */

    // This method is no longer needed as we read from a single CSV
    /*
    private Dataset<Row> readLyrics(String inputDirectory, String path) {
        // ... original code ...
    }
    */
    // --- END OF REFACTORED DATA READING ---


    private Genre getGenre(Double value) {
        for (Genre genre: Genre.values()){
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }
        // If the model predicts something other than 0.0 or 1.0, return UNKNOWN
        System.out.println("Warning: Model predicted an unknown label value: " + value);
        return Genre.UNKNOWN;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        // Handle potential case where model hasn't been trained yet or failed
        if (model.avgMetrics() != null && model.avgMetrics().length > 0) {
             Arrays.sort(model.avgMetrics());
             modelStatistics.put("Best model metrics (higher is better, e.g., AreaUnderROC)", model.avgMetrics()[model.avgMetrics().length - 1]);
        } else {
             modelStatistics.put("Best model metrics", "N/A (Model training might have failed or metrics unavailable)");
        }

        return modelStatistics;
    }

    void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

    void saveModel(CrossValidatorModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    void saveModel(PipelineModel model, String modelOutputDirectory) {
        this.mlService.saveModel(model, modelOutputDirectory);
    }

    // Keep setters for testability if needed
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