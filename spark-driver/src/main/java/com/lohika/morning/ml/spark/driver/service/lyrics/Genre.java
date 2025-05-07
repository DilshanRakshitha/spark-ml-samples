package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {

    // Original
    METAL("Metal \\m//", 0.0),
    POP("Pop <(^.^)/", 1.0),

    // New genres for multi-class
    COUNTRY("Country", 2.0),
    BLUES("Blues", 3.0),
    JAZZ("Jazz", 4.0),
    REGGAE("Reggae", 5.0),
    ROCK("Rock", 6.0),
    HIP_HOP("Hip Hop", 7.0), // Note: CSV has "hip hop", we'll handle case in loading

    UNKNOWN("Don't know :(", -1.0); // Keep for unclassified or during prediction

    private final String name;
    private final Double value;

    Genre(final String name, final Double value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Double getValue() {
        return value;
    }

    // Corrected method declaration
    public static Genre fromValue(Double value) {
        if (value == null) { // Added null check for robustness
            return UNKNOWN;
        }
        for (Genre genre : Genre.values()) {
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }
        return UNKNOWN;
    }

    public static Genre fromString(String text) {
        if (text == null || text.trim().isEmpty()) { // Added null/empty check
            return UNKNOWN;
        }
        String lowerText = text.toLowerCase().trim();
        // Handle "hip hop" specifically if it's stored with a space
        if ("hip hop".equals(lowerText)) {
            return HIP_HOP;
        }
        for (Genre genre : Genre.values()) {
            // Try matching enum name (e.g. "HIP_HOP" for "hip_hop")
            if (genre.name().equalsIgnoreCase(lowerText.replace(" ", "_"))) {
                return genre;
            }
            // Try matching display name (e.g. "Hip Hop" for "Hip Hop")
            if (genre.getName().equalsIgnoreCase(text.trim())) {
                return genre;
            }
        }
        return UNKNOWN;
    }
}