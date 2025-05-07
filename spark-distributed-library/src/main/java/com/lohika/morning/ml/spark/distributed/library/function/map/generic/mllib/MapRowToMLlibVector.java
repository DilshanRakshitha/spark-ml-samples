// spark-distributed-library/src/main/java/com/lohika/morning/ml/spark/distributed/library/function/map/generic/mllib/MapRowToMLlibVector.java
package com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector; // mllib Vector
import org.apache.spark.mllib.linalg.Vectors; // mllib Vectors utility
import org.apache.spark.sql.Row;

public class MapRowToMLlibVector implements Function<Row, Vector> {

    @Override
    public Vector call(Row inputRow) throws Exception {
        // Assumes 'features' field is org.apache.spark.ml.linalg.Vector in the Row
        org.apache.spark.ml.linalg.Vector mlFeatures = inputRow.getAs("features");

        // Convert from ml.linalg.Vector to mllib.linalg.Vector
        return Vectors.fromML(mlFeatures);
    }
}