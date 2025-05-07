package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column;
import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;

import java.util.ArrayList; // Import ArrayList
import java.util.Arrays;
import java.util.List; // Import List
import java.util.Map;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.springframework.stereotype.Component;

@Component("FeedForwardNeuralNetworkPipeline")
public class FeedForwardNeuralNetworkPipeline extends CommonLyricsPipeline {

    @Override
    public CrossValidatorModel classify() {
        Dataset<Row> sentences = readLyrics(); // This sets this.numClasses

        if (this.numClasses < 2) {
             throw new IllegalStateException("FeedForwardNeuralNetworkPipeline requires at least 2 classes for training. Found: " + this.numClasses);
        }

        Cleanser cleanser = new Cleanser();
        Numerator numerator = new Numerator();
        Tokenizer tokenizer = new Tokenizer().setInputCol(CLEAN.getName()).setOutputCol(WORDS.getName());
        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol(WORDS.getName()).setOutputCol(FILTERED_WORDS.getName());
        Exploder exploder = new Exploder();
        Stemmer stemmer = new Stemmer();
        Uniter uniter = new Uniter();
        Verser verser = new Verser();
        Word2Vec word2Vec = new Word2Vec().setInputCol(Column.VERSE.getName()).setOutputCol("features").setMinCount(0);


        MultilayerPerceptronClassifier multilayerPerceptronClassifier = new MultilayerPerceptronClassifier()
                .setBlockSize(128)
                .setSeed(1234L);


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
                        multilayerPerceptronClassifier});

        int fixedW2VVectorSize = 100;

        // *** Build grid for parameters *other* than layers ***
        ParamMap[] baseParamGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{8, 16})
                .addGrid(word2Vec.vectorSize(), new int[] {fixedW2VVectorSize})
                .addGrid(multilayerPerceptronClassifier.maxIter(), new int[] {50, 100})
                .build();

        // *** Manually add the layers configurations ***
        List<ParamMap> finalParamMaps = new ArrayList<>();
        int[][] layerOptions = {
            new int[]{fixedW2VVectorSize, 50, this.numClasses},
            new int[]{fixedW2VVectorSize, 30, this.numClasses}
        };

        for (ParamMap baseMap : baseParamGrid) {
            for (int[] layers : layerOptions) {
                // Create a copy and add the specific layers setting
                finalParamMaps.add(baseMap.copy().put(multilayerPerceptronClassifier.layers(), layers));
            }
        }

        ParamMap[] paramGrid = finalParamMaps.toArray(new ParamMap[0]); // Convert List<ParamMap> back to ParamMap[]


        MulticlassClassificationEvaluator evaluator = getAccuracyEvaluator();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid) // Use the manually constructed grid
                .setNumFolds(3);

        System.out.println("Starting training for FeedForward Neural Network with Cross-Validation...");
        CrossValidatorModel model = crossValidator.fit(sentences);
        System.out.println("Training finished for FeedForward Neural Network.");

        saveModel(model, getModelDirectory());

        return model;
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        MultilayerPerceptronClassificationModel mlpcModel = (MultilayerPerceptronClassificationModel) stages[stages.length - 1];

        modelStatistics.put("MLPC Layers (Actual used in best model)", Arrays.toString(mlpcModel.layers()));

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/feed-forward-neural-network_multiclass/";
    }
}