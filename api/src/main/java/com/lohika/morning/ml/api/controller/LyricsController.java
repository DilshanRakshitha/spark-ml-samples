package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.api.service.LyricsService;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.util.Map;
import java.util.HashMap; // <<< ADD THIS IMPORT
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/lyrics")
public class LyricsController {

    private static final Logger log = LoggerFactory.getLogger(LyricsController.class);

    @Autowired
    private LyricsService lyricsService;

    @RequestMapping(value = "/train", method = RequestMethod.GET)
    ResponseEntity<Map<String, Object>> trainLyricsModel(
            @RequestParam(value = "modelName", required = true) String modelName) {
        try {
            log.info("Received /train request for model: {}", modelName);
            Map<String, Object> trainStatistics = lyricsService.classifyLyrics(modelName);
            return new ResponseEntity<>(trainStatistics, HttpStatus.OK);
        } catch (IllegalArgumentException e) {
             log.error("Bad request for training model {}: {}", modelName, e.getMessage());
             Map<String, Object> errorResponse = new HashMap<>(); // Now HashMap is recognized
             errorResponse.put("error", e.getMessage());
             return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
        } catch (Exception e) {
            log.error("Internal server error training model {}: {}", modelName, e.getMessage(), e);
            Map<String, Object> errorResponse = new HashMap<>(); // Now HashMap is recognized
            errorResponse.put("error", "Internal server error during training.");
            return new ResponseEntity<>(errorResponse, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @RequestMapping(value = "/predict", method = RequestMethod.POST)
    ResponseEntity<GenrePrediction> predictGenre(
            @RequestParam(value = "modelName", required = true) String modelName,
            @RequestBody String unknownLyrics) {
         try {
            log.info("Received /predict request for model: {}", modelName);
            if (unknownLyrics == null || unknownLyrics.trim().isEmpty()) {
                 throw new IllegalArgumentException("Lyrics body cannot be empty.");
            }
            GenrePrediction genrePrediction = lyricsService.predictGenre(modelName, unknownLyrics);
            return new ResponseEntity<>(genrePrediction, HttpStatus.OK);
         } catch (IllegalArgumentException e) {
             log.error("Bad request for predicting with model {}: {}", modelName, e.getMessage());
             GenrePrediction errorPrediction = new GenrePrediction("Error: " + e.getMessage());
             return new ResponseEntity<>(errorPrediction, HttpStatus.BAD_REQUEST);
         } catch (Exception e) {
            log.error("Internal server error predicting with model {}: {}", modelName, e.getMessage(), e);
            GenrePrediction errorPrediction = new GenrePrediction("Error: Internal Server Error");
            return new ResponseEntity<>(errorPrediction, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}