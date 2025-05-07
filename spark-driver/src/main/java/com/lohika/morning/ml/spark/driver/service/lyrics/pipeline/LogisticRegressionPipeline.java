package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import static com.lohika.morning.ml.spark.distributed.library.function.map.lyrics.Column.*;
import com.lohika.morning.ml.spark.driver.service.lyrics.transformer.*;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
// import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator; // Remove this
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator; // Add this
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
import org.springframework.stereotype.Component;

@Component("LogisticRegressionPipeline")
public class LogisticRegressionPipeline extends CommonLyricsPipeline {

    @Override
    public CrossValidatorModel classify() {
        Dataset<Row> sentences = readLyrics(); // This will now load multi-class data and set numClasses

        Cleanser cleanser = new Cleanser();
        Numerator numerator = new Numerator();
        Tokenizer tokenizer = new Tokenizer().setInputCol(CLEAN.getName()).setOutputCol(WORDS.getName());
        StopWordsRemover stopWordsRemover = new StopWordsRemover().setInputCol(WORDS.getName()).setOutputCol(FILTERED_WORDS.getName());
        Exploder exploder = new Exploder();
        Stemmer stemmer = new Stemmer();
        Uniter uniter = new Uniter();
        Verser verser = new Verser();
        Word2Vec word2Vec = new Word2Vec().setInputCol(VERSE.getName()).setOutputCol("features").setMinCount(0);
        LogisticRegression logisticRegression = new LogisticRegression(); // LogisticRegression supports multi-class out-of-the-box

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
                        logisticRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8, 16})
                .addGrid(word2Vec.vectorSize(), new int[] {50})
                .addGrid(logisticRegression.regParam(), new double[] {0.01, 0.1}) // Added another regParam for example
                .addGrid(logisticRegression.maxIter(), new int[] {30}) // Adjusted maxIter
                .build();

        // Use the common MulticlassClassificationEvaluator for accuracy
        MulticlassClassificationEvaluator evaluator = getAccuracyEvaluator();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator) // Use the multiclass evaluator
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(2); // Reduced folds for faster testing, increase for better results (e.g., 5 or 10)

        System.out.println("Starting training for Logistic Regression with Cross-Validation...");
        CrossValidatorModel model = crossValidator.fit(sentences);
        System.out.println("Training finished for Logistic Regression.");

        saveModel(model, getModelDirectory());
        
        // The getModelStatistics in CommonLyricsPipeline will be called by the service
        // but we can print specific details here if needed.
        System.out.println("Logistic Regression Model specific statistics:");
        Map<String, Object> detailedStats = getModelStatistics(model); // This already prints general stats
        // Add any LogisticRegression specific stats to 'detailedStats' if you want here
        // For example, coefficients if meaningful for multi-class (it will be a matrix)
        PipelineModel bestPipeline = (PipelineModel) model.bestModel();
        LogisticRegressionModel lrModel = (LogisticRegressionModel) bestPipeline.stages()[bestPipeline.stages().length -1];
        System.out.println("Logistic Regression Coefficients: " + lrModel.coefficientMatrix());
        System.out.println("Logistic Regression Intercepts: " + lrModel.interceptVector());


        return model;
    }

    // This override can now add more specific details or just rely on the common one
    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = super.getModelStatistics(model); // Gets accuracy and best params

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Transformer[] stages = bestModel.stages();

        // These are already in the best params from super.getModelStatistics
        // modelStatistics.put("Sentences in verse", ((Verser) stages[7]).getSentencesInVerse());
        // modelStatistics.put("Word2Vec vocabulary", ((Word2VecModel) stages[8]).getVectors().count());
        // modelStatistics.put("Vector size", ((Word2VecModel) stages[8]).getVectorSize());
        
        // LogisticRegressionModel specific parameters
        LogisticRegressionModel lrActualModel = (LogisticRegressionModel) stages[stages.length - 1]; // Last stage is LR
        modelStatistics.put("LR Reg parameter", lrActualModel.getRegParam());
        modelStatistics.put("LR Max iterations", lrActualModel.getMaxIter());
        modelStatistics.put("LR Family", lrActualModel.getFamily()); // multinomial for multi-class

        // No need to call printModelStatistics here, the service/controller will do it
        // printModelStatistics(modelStatistics); 

        return modelStatistics;
    }

    @Override
    protected String getModelDirectory() {
        return getLyricsModelDirectoryPath() + "/logistic-regression_multiclass/"; // Changed path
    }
}