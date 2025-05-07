package com.lohika.morning.ml.spark.driver.service;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test; // Ensure this is org.junit.Test for JUnit 4 if BaseTest uses SpringJUnit4ClassRunner
// Or org.junit.jupiter.api.Test for JUnit 5 (if BaseTest is updated)

public class TFIDFTest extends BaseTest { // BaseTest uses SpringJUnit4ClassRunner

    @Test
    public void test() {
        List<Row> data = Arrays.asList(
                RowFactory.create(0.0, "Hi I heard about Spark"),
                RowFactory.create(0.0, "I wish Java could use case classes"),
                RowFactory.create(1.0, "Logistic regression models are neat")
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
        });
        Dataset<Row> sentenceData = getSparkSession().createDataFrame(data, schema);

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        System.out.println("Words:\n");
        wordsData.select("label", "words").show(false); // Using show for better readability
        System.out.println("----------------");

        int numFeatures = 20; // Increased for more realistic hashing
        HashingTF hashingTF = new HashingTF()
                .setInputCol("words") // Corrected from "sentence"
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);
        System.out.println("Raw Features (TF):\n");
        featurizedData.select("label", "rawFeatures").show(false);
        System.out.println("----------------");

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);

        System.out.println("Features (TF-IDF):\n");
        rescaledData.select("label", "features").show(false);
        System.out.println("----------------");
    }
}