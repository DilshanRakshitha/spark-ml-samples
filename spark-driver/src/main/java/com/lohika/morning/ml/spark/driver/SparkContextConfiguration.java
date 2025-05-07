package com.lohika.morning.ml.spark.driver;

import org.apache.spark.SparkConf;
// import org.apache.spark.SparkContext; // No longer directly creating SparkContext first
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;

@Configuration
@PropertySource("classpath:spark.properties")
@ComponentScan("com.lohika.morning.ml.spark.driver.*")
public class SparkContextConfiguration {

    @Value("${spark.master}")
    private String master;

    @Value("${spark.application-name}")
    private String applicationName;

    @Value("${spark.distributed-libraries}")
    private String[] distributedLibraries;

    @Value("${spark.cores.max:}") // Allow empty, Spark will use default
    private String coresMax;

    @Value("${spark.driver.memory}")
    private String driverMemory;

    @Value("${spark.executor.memory}")
    private String executorMemory;

    @Value("${spark.serializer}")
    private String serializer;

    @Value("${spark.sql.shuffle.partitions}")
    private String sqlShufflePartitions;

    @Value("${spark.default.parallelism}")
    private String defaultParallelism;

    @Value("${spark.kryoserializer.buffer.max}")
    private String kryoserializerBufferMax;

    @Bean
    public SparkSession sparkSession() {
        SparkSession.Builder builder = SparkSession.builder()
                .master(master)
                .appName(applicationName)
                .config("spark.driver.memory", driverMemory)
                .config("spark.executor.memory", executorMemory)
                .config("spark.serializer", serializer)
                .config("spark.kryoserializer.buffer.max", kryoserializerBufferMax)
                .config("spark.kryo.registrationRequired", "false")
                .config("spark.sql.shuffle.partitions", sqlShufflePartitions)
                .config("spark.default.parallelism", defaultParallelism);

        if (distributedLibraries != null && distributedLibraries.length > 0 && !distributedLibraries[0].isEmpty()) {
            // Filter out empty strings that might come from placeholder resolution
            String validJars = java.util.Arrays.stream(distributedLibraries)
                    .filter(jar -> jar != null && !jar.trim().isEmpty() && !jar.contains("<") && !jar.contains(">"))
                    .collect(java.util.stream.Collectors.joining(","));
            if (!validJars.isEmpty()) {
                builder.config("spark.jars", validJars);
            }
        }

        if (coresMax != null && !coresMax.isEmpty()) {
            builder.config("spark.cores.max", coresMax);
        }

        // For local mode, spark.jars might not be needed or could cause issues if path is invalid
        // SparkConf handles setJars with an array, SparkSession builder expects comma-separated string for "spark.jars"
        // Or programmatically add jars to SparkContext before creating SparkSession (more complex now)

        return builder.getOrCreate();
    }

    @Bean
    // Static is extremely important here.
    // It should be created before @Configuration as it is also component.
    public static PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer() {
        return new PropertySourcesPlaceholderConfigurer();
    }

}