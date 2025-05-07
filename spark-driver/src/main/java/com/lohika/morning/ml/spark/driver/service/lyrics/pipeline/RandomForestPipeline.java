package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;

import java.util.Arrays; // Added import
import java.util.Map;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator; // Changed import
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row; // Added import
import org.springframework.stereotype.Component;

@Component("RandomForestPipeline")
public class RandomForestPipeline extends CommonLyricsPipeline {

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

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{8, 16}) // Example
                .addGrid(word2Vec.vectorSize(), new int[] {100, 200}) // Example
                .addGrid(randomForest.numTrees(), new int[] {10, 20})    // Example tuning
                .addGrid(randomForest.maxDepth(), new int[] {5, 10})     // Example tuning
                .addGrid(randomForest.maxBins(), new int[] {32})       // Example tuning
                .build();

        // Use the common multiclass accuracy evaluator
        MulticlassClassificationEvaluator evaluator = getAccuracyEvaluator();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator) // Use multiclass evaluator
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3); // Reduced folds for faster testing

        System.out.println("Starting training for Random Forest with Cross-Validation...");
        CrossValidatorModel model = crossValidator.fit(sentences);
        System.out.println("Training finished for Random Forest.");

        saveModel(model, getModelDirectory());

        return model;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model); // Gets accuracy and best params

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        // Extract specific parameters from the best model
        // Word2VecModel word2VecModel = (Word2VecModel) stages[8]; // Info available in "Best Parameters"
        RandomForestClassificationModel rfModel = (RandomForestClassificationModel) stages[9]; // Assuming RF is stage 9

        // These might already be in "Best Parameters", but can be explicitly added
        // modelStatistics.put("Sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        // modelStatistics.put("Vector size", word2VecModel.getVectorSize());
        modelStatistics.put("RF Num Trees (Best Model)", rfModel.getNumTrees());
        modelStatistics.put("RF Max Depth (Best Model)", rfModel.getMaxDepth());
        modelStatistics.put("RF Max Bins (Best Model)", rfModel.getMaxBins());
        modelStatistics.put("RF Feature Subset Strategy (Best Model)", rfModel.getFeatureSubsetStrategy());

        // printModelStatistics(modelStatistics); // Called by service

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
         // Changed path
        return getLyricsModelDirectoryPath() + "/random-forest_multiclass/";
    }
}