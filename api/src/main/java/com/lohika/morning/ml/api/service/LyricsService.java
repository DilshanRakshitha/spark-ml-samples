package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import com.lohika.morning.ml.spark.driver.service.lyrics.pipeline.LyricsPipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.NoSuchBeanDefinitionException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value; // <<< ADD THIS IMPORT
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

// Removed javax.annotation.Resource as @Value is used
import javax.annotation.PostConstruct;
// import javax.annotation.Resource;
import java.util.*;
import java.util.stream.Collectors;
import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;


@Component
public class LyricsService {

    private static final Logger log = LoggerFactory.getLogger(LyricsService.class);

    @Autowired
    private ApplicationContext applicationContext;

    @Autowired
    private SparkSession sparkSession;

    private final List<String> availablePipelines = Arrays.asList(
            "LogisticRegressionPipeline",
            "FeedForwardNeuralNetworkPipeline",
            "NaiveBayesBagOfWordsPipeline",
            "NaiveBayesTFIDFPipeline",
            "RandomForestPipeline"
    );

    // Using @Value for the default pipeline name
    @Value("${lyrics.pipeline}")
    private String defaultPipelineName;


    private LyricsPipeline getPipeline(String modelName) {
        try {
            if (modelName == null || modelName.isEmpty() || !availablePipelines.contains(modelName)) {
                log.warn("Invalid or missing modelName '{}'. Using default: {}", modelName, defaultPipelineName);
                modelName = defaultPipelineName;
            }
            log.info("Using pipeline bean: {}", modelName);
            return applicationContext.getBean(modelName, LyricsPipeline.class);
        } catch (NoSuchBeanDefinitionException e) {
            log.error("Error getting pipeline bean '{}'", modelName, e);
            throw new IllegalArgumentException("Invalid model name specified: " + modelName);
        }
    }

    public Map<String, Object> classifyLyrics(String modelName) {
        log.info("Received request to train model: {}", modelName);
        LyricsPipeline pipeline = getPipeline(modelName);
        CrossValidatorModel model = pipeline.classify();
        Map<String, Object> stats = pipeline.getModelStatistics(model);
        stats.put("trainedModel", modelName);
        return stats;
    }

    public GenrePrediction predictGenre(final String modelName, final String unknownLyrics) {
         log.info("Received prediction request for model '{}'", modelName);
         LyricsPipeline pipeline = getPipeline(modelName);

         try {
             String lyrics[] = unknownLyrics.split("\\r?\\n");
             Dataset<String> lyricsDataset = sparkSession.createDataset(Arrays.asList(lyrics), Encoders.STRING());

             Dataset<Row> unknownLyricsDataset = lyricsDataset
                     .withColumnRenamed("value", VALUE.getName())
                     .withColumn(LABEL.getName(), functions.lit(Genre.UNKNOWN.getValue()))
                     .withColumn(ID.getName(), functions.monotonically_increasing_id().cast(DataTypes.StringType));

             CrossValidatorModel cvModel = pipeline.loadModel();
             PipelineModel bestModel = (PipelineModel) cvModel.bestModel();

             Dataset<Row> predictionsDataset = bestModel.transform(unknownLyricsDataset);

             // *** Fix: Changed isEmpty() to count() == 0 ***
             if (predictionsDataset.count() == 0) {
                log.warn("Prediction resulted in empty dataset for model {}", modelName);
                return new GenrePrediction(Genre.UNKNOWN.getName());
             }

             Row predictionRow = predictionsDataset.first();

             Double predictionLabel = predictionRow.getAs("prediction");
             Genre predictedGenre = Genre.fromValue(predictionLabel);
             Map<String, Double> probabilityMap = new HashMap<>(); // Initialize map here

             if (Arrays.asList(predictionsDataset.columns()).contains("probability")) {
                 Vector probabilityVector = predictionRow.getAs("probability");
                 for (Genre genre : Genre.values()) {
                     if (genre != Genre.UNKNOWN) {
                         int index = genre.getValue().intValue();
                         if (index >= 0 && index < probabilityVector.size()) {
                             probabilityMap.put(genre.getName(), probabilityVector.apply(index));
                         }
                     }
                 }
                 log.debug("Probabilities calculated: {}", probabilityMap);
             } else {
                 log.warn("Probability column not found in prediction output for model {}", modelName);
             }

            return new GenrePrediction(predictedGenre.getName(), probabilityMap.isEmpty() ? null : probabilityMap);

         } catch (Exception e) {
              log.error("Error during prediction for model '{}'", modelName, e);
              return new GenrePrediction(Genre.UNKNOWN.getName());
         }
    }
}