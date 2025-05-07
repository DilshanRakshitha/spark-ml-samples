package com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector; // mllib Vector
import org.apache.spark.mllib.linalg.Vectors; // mllib Vectors
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class MapRowToMLlibLabeledPoint implements Function<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        // Assuming 'label' is Double and 'features' is org.apache.spark.ml.linalg.Vector in the Row
        Double label = inputRow.getAs("label");
        org.apache.spark.ml.linalg.Vector mlFeatures = inputRow.getAs("features");

        // Convert from ml.linalg.Vector to mllib.linalg.Vector
        Vector mllibFeatures = Vectors.fromML(mlFeatures);

        return new LabeledPoint(label, mllibFeatures);
    }
}