package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.util.Map;
import org.apache.spark.ml.tuning.CrossValidatorModel;

public interface LyricsPipeline {

    // Trains the model and returns it
    CrossValidatorModel classify();

    // Removed: Prediction logic is now primarily in LyricsService
    // GenrePrediction predict(String unknownLyrics);

    // Gets stats from a trained model instance
    Map<String, Object> getModelStatistics(CrossValidatorModel model);

    // Method to load the pre-trained model
    CrossValidatorModel loadModel();
}