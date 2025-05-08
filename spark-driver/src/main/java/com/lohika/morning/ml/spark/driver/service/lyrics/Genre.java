package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.util.Arrays;

public enum Genre {
    UNKNOWN(-1.0, "Unknown"), // Or a value like 8.0 if you want it last and separate
    POP(0.0, "Pop"),
    ROCK(1.0, "Rock"),
    COUNTRY(2.0, "Country"),
    JAZZ(3.0, "Jazz"),
    BLUES(4.0, "Blues"),
    REGGAE(5.0, "Reggae"),
    HIP_HOP(6.0, "HipHop"), // Assuming this was 0-6
    DANCE(7.0, "Dance");    // New 8th class, value 7.0

    private Double value;
    private String name;

    Genre(Double value, String name) {
        this.value = value;
        this.name = name;
    }

    public Double getValue() {
        return value;
    }

    public String getName() {
        return name;
    }

    public static Genre fromValue(final Double value) {
        return Arrays.stream(Genre.values())
                .filter(genre -> genre.getValue().equals(value))
                .findFirst()
                .orElse(UNKNOWN);
    }

    public static Genre fromName(final String name) {
        return Arrays.stream(Genre.values())
                .filter(genre -> genre.getName().equalsIgnoreCase(name))
                .findFirst()
                .orElse(UNKNOWN);
    }

    /**
     * Gets all genres that can be predicted (i.e., not UNKNOWN).
     * @return An array of predictable Genre enums.
     */
    public static Genre[] getPredictableGenres() {
        return Arrays.stream(Genre.values())
                     .filter(genre -> !genre.equals(UNKNOWN))
                     .toArray(Genre[]::new);
    }

    /**
     * Converts a genre name string to its numerical value.
     * This is used by the Numerator transformer to create the initial label column.
     * Ensure this mapping is consistent with how StringIndexer will treat labels.
     * @param genreName The string name of the genre.
     * @return The double value of the genre.
     */
    public static double stringToValue(String genreName) {
        return fromName(genreName).getValue();
    }
}