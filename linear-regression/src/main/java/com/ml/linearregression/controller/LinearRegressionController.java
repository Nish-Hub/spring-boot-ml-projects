package com.ml.linearregression.controller;


import com.ml.linearregression.service.LinearRegressionService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/linear-regression")
public class LinearRegressionController {
    private final LinearRegressionService linearRegressionService;

    public LinearRegressionController(LinearRegressionService linearRegressionService) {
        this.linearRegressionService = linearRegressionService;
    }

    @PostMapping("/train")
    public String trainModel() {
        linearRegressionService.trainModel();
        return "Model trained successfully!";
    }

    @GetMapping("/model")
    public String getModel() {
        return linearRegressionService.getModel();
    }

    @GetMapping("/predict")
    public double predict(@RequestParam double x) {
        return linearRegressionService.predict(x);
    }
}
