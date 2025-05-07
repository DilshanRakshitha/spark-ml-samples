package com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector; // mllib Vector
import org.apache.spark.mllib.linalg.Vectors; // mllib Vectors
import org.apache.spark.sql.Row;

public class MapRowToMLlibVector implements Function<Row, Vector> {

    @Override
    public Vector call(Row inputRow) throws Exception {
        // Assuming 'features' is org.apache.spark.ml.linalg.Vector in the Row
        org.apache.spark.ml.linalg.Vector mlFeatures = inputRow.getAs("features");

        // Convert from ml.linalg.Vector to mllib.linalg.Vector
        return Vectors.fromML(mlFeatures);
    }
}