package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;

import java.util.Arrays;
import java.util.Map;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component("RandomForestPipeline")
public class RandomForestPipeline extends CommonLyricsPipeline {

    private static final Logger log = LoggerFactory.getLogger(RandomForestPipeline.class);

    @Override
    public CrossValidatorModel classify() {
        Dataset<Row> sentences = readLyrics(); // Reads multi-class data

        Cleanser cleanser = new Cleanser();
        Numerator numerator = new Numerator();
        Tokenizer tokenizer = new Tokenizer().setInputCol(CLEAN.getName()).setOutputCol(WORDS.getName());
        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol(WORDS.getName()).setOutputCol(FILTERED_WORDS.getName());
        Exploder exploder = new Exploder();
        Stemmer stemmer = new Stemmer();
        Uniter uniter = new Uniter();
        Verser verser = new Verser();
        Word2Vec word2Vec = new Word2Vec().setInputCol(VERSE.getName()).setOutputCol("features").setMinCount(0);
        RandomForestClassifier randomForest = new RandomForestClassifier(); // Supports multi-class natively

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        cleanser,
                        numerator,
                        tokenizer,
                        stopWordsRemover,
                        exploder,
                        stemmer,
                        uniter,
                        verser,
                        word2Vec,
                        randomForest});

        // Reduced Parameter Grid for faster training / lower memory
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{8})        // Reduced options
                .addGrid(word2Vec.vectorSize(), new int[] {50, 100})    // Reduced options
                .addGrid(randomForest.numTrees(), new int[] {10, 15})     // Reduced number of trees
                .addGrid(randomForest.maxDepth(), new int[] {4, 6})      // Reduced max depth significantly
                .addGrid(randomForest.maxBins(), new int[] {32})        // Kept default, could reduce to 16 if needed
                .build();

        // Use the common multiclass accuracy evaluator
        MulticlassClassificationEvaluator evaluator = getAccuracyEvaluator();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator) // Use multiclass evaluator
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(2); // Reduced folds drastically for testing

        System.out.println("Starting training for Random Forest (REDUCED SETTINGS) with Cross-Validation...");
        // Add repartitioning before fit to potentially help with memory per task
        int numPartitions = sparkSession.sparkContext().defaultParallelism() * 2;
        log.info("Repartitioning data before fit to {} partitions", numPartitions);
        sentences = sentences.repartition(numPartitions);

        CrossValidatorModel model = crossValidator.fit(sentences);
        System.out.println("Training finished for Random Forest (REDUCED SETTINGS).");

        // *** Changed: Save to the standard directory ***
        saveModel(model, getModelDirectory());
        // *** End Change ***

        return model;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model); // Gets accuracy and best params

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) stages[stages.length - 1]; // Assuming RF is last stage

        modelStatistics.put("RF Num Trees (Best Model)", rfModel.getNumTrees());
        modelStatistics.put("RF Max Depth (Best Model)", rfModel.getMaxDepth());
        modelStatistics.put("RF Max Bins (Best Model)", rfModel.getMaxBins());
        modelStatistics.put("RF Feature Subset Strategy (Best Model)", rfModel.getFeatureSubsetStrategy());

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
         // Path for the multiclass Random Forest model
        return getLyricsModelDirectoryPath() + "/random-forest_multiclass/";
    }
}