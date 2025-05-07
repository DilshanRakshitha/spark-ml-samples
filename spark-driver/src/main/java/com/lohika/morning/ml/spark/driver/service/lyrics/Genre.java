package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {

    // *** CHANGE HERE: Removed symbols from display name ***
    METAL("Metal", 0.0),
    POP("Pop", 1.0),
    // *** END CHANGE ***

    // New genres for multi-class
    COUNTRY("Country", 2.0),
    BLUES("Blues", 3.0),
    JAZZ("Jazz", 4.0),
    REGGAE("Reggae", 5.0),
    ROCK("Rock", 6.0),
    HIP_HOP("Hip Hop", 7.0),

    UNKNOWN("Don't know :(", -1.0);

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

    public static Genre fromValue(Double value) {
        if (value == null) {
            return UNKNOWN;
        }
        for (Genre genre : Genre.values()) {
            // Use tolerance for double comparison if needed, but equals should work for these specific values
             if (Math.abs(genre.getValue() - value) < 0.0001) {
                return genre;
            }
        }
         System.out.println("Warning: Could not map prediction value " + value + " back to a known Genre.");
        return UNKNOWN;
    }

    public static Genre fromString(String text) {
        if (text == null || text.trim().isEmpty()) {
            return UNKNOWN;
        }
        String lowerText = text.toLowerCase().trim();
        if ("hip hop".equals(lowerText)) {
            return HIP_HOP;
        }
        for (Genre genre : Genre.values()) {
            if (genre.name().equalsIgnoreCase(lowerText.replace(" ", "_"))) {
                return genre;
            }
             // Also check the (now cleaned) display name directly
            if (genre.getName().equalsIgnoreCase(text.trim())) {
                return genre;
            }
        }
        return UNKNOWN;
    }
}