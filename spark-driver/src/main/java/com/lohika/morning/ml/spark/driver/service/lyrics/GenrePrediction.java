package com.lohika.morning.ml.spark.driver.service.lyrics;

// Add import if needed for List/Map if returning multiple probabilities with names
import java.util.Map;

public class GenrePrediction {

    private String genre; // Top predicted genre name
    private Map<String, Double> probabilities; // Map genre name to probability

    // Constructor used when probabilities are available
    public GenrePrediction(String genre, Map<String, Double> probabilities) {
        this.genre = genre;
        this.probabilities = probabilities;
    }

    // Constructor used when probabilities might not be available (or not needed)
    public GenrePrediction(String genre) {
        this.genre = genre;
        this.probabilities = null; // Explicitly set to null
    }

    public String getGenre() {
        return genre;
    }

    public Map<String, Double> getProbabilities() {
        return probabilities;
    }

    // No-arg constructor for frameworks like Jackson
    public GenrePrediction() {}
}