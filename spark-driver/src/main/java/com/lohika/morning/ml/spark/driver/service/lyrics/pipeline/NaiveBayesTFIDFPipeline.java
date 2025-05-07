package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;

import java.util.Arrays; // Added import
import java.util.Map;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel; // Added import
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator; // Changed import
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row; // Added import
import org.springframework.stereotype.Component;

@Component("NaiveBayesTFIDFPipeline")
public class NaiveBayesTFIDFPipeline extends CommonLyricsPipeline {

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
        HashingTF tf = new HashingTF().setInputCol(VERSE.getName()).setOutputCol("rawFeatures");
        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        NaiveBayes naiveBayes = new NaiveBayes().setModelType("multinomial");

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
                        tf,
                        idf,
                        naiveBayes});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16, 32})
                .addGrid(tf.numFeatures(), new int[]{10000, 20000}) // Example tuning
                .addGrid(idf.minDocFreq(), new int[]{0, 1, 2})
                .addGrid(naiveBayes.smoothing(), new double[]{0.5, 1.0}) // Example tuning
                .build();

        // Use the common multiclass accuracy evaluator
        MulticlassClassificationEvaluator evaluator = getAccuracyEvaluator();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator) // Use multiclass evaluator
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3); // Reduced folds for faster testing

        System.out.println("Starting training for Naive Bayes (TF-IDF) with Cross-Validation...");
        CrossValidatorModel model = crossValidator.fit(sentences);
        System.out.println("Training finished for Naive Bayes (TF-IDF).");

        saveModel(model, getModelDirectory());

        return model;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model); // Gets accuracy and best params

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        // Extract specific parameters from the best model
        HashingTF tfModel = (HashingTF) stages[8];          // Assuming HashingTF is stage 8
        IDFModel idfModel = (IDFModel) stages[9];            // Assuming IDF is stage 9
        NaiveBayesModel nbModel = (NaiveBayesModel) stages[10]; // Assuming NaiveBayes is stage 10

        // These might already be in "Best Parameters", but can be explicitly added
        // modelStatistics.put("Sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        modelStatistics.put("TF Num Features (Best Model)", tfModel.getNumFeatures());
        modelStatistics.put("IDF Min Doc Frequency (Best Model)", idfModel.getMinDocFreq());
        modelStatistics.put("NB Smoothing (Best Model)", nbModel.getSmoothing());
        modelStatistics.put("NB Model Type (Best Model)", nbModel.getModelType());

        // printModelStatistics(modelStatistics); // Called by service

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        // Changed path
        return getLyricsModelDirectoryPath() + "/naive-bayes-tfidf_multiclass/";
    }
}