package com.lohika.morning.ml.spark.driver;

// import org.apache.spark.SparkContext; // No longer directly creating SparkContext first
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;
import java.util.Arrays;
import java.util.stream.Collectors;

@Configuration
@PropertySource("classpath:spark.properties")
@ComponentScan("com.lohika.morning.ml.spark.driver.*")
public class SparkContextConfiguration {

    @Value("${spark.master}")
    private String master;

    @Value("${spark.application-name}")
    private String applicationName;

    @Value("${spark.distributed-libraries:}") // Default to empty string if not present
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
                .config("spark.kryo.registrationRequired", "false") // Common Kryo setting
                .config("spark.sql.shuffle.partitions", sqlShufflePartitions)
                .config("spark.default.parallelism", defaultParallelism);

        if (distributedLibraries != null && distributedLibraries.length > 0) {
            String validJars = Arrays.stream(distributedLibraries)
                    .filter(jar -> jar != null && !jar.trim().isEmpty() && !jar.contains("<should be")) // Filter out placeholders
                    .collect(Collectors.joining(","));
            if (!validJars.isEmpty()) {
                builder.config("spark.jars", validJars);
            }
        }

        if (coresMax != null && !coresMax.isEmpty()) {
            builder.config("spark.cores.max", coresMax);
        }

        return builder.getOrCreate();
    }

    @Bean
    public static PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer() {
        return new PropertySourcesPlaceholderConfigurer();
    }
}