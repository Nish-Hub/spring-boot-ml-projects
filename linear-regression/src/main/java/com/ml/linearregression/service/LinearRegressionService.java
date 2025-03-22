package com.ml.linearregression.service;

import org.apache.commons.csv.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

@Service
public class LinearRegressionService {
    private double m = 0; // Slope
    private double b = 0; // Intercept
    private final double learningRate = 0.01;
    private final int epochs = 1000;
    private final String csvFilePath = "data.csv"; // Path to CSV file

    public void trainModel() {
        double[][] data = readCSV(csvFilePath,true);
        double[] X = data[0];
        double[] Y = data[1];

        int n = X.length;

        for (int i = 0; i < epochs; i++) {
            double mGradient = 0;
            double bGradient = 0;

            for (int j = 0; j < n; j++) {
                double prediction = (m * X[j]) + b;
                mGradient += -(2.0 / n) * X[j] * (Y[j] - prediction);
                bGradient += -(2.0 / n) * (Y[j] - prediction);
            }

            m -= learningRate * mGradient;
            b -= learningRate * bGradient;
        }
    }

    public double predict(double x) {
        return m * x + b;
    }

    public String getModel() {
        return "Y = " + m + "X + " + b;
    }

    private double[][] readCSV(String filename,boolean isClasspathResource) {
        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();

        try (InputStream inputStream = new ClassPathResource(filename).getInputStream();
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            for (CSVRecord record : csvParser) {
                xList.add(Double.parseDouble(record.get("X")));
                yList.add(Double.parseDouble(record.get("Y")));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] X = xList.stream().mapToDouble(Double::doubleValue).toArray();
        double[] Y = yList.stream().mapToDouble(Double::doubleValue).toArray();
        return new double[][]{X, Y};
    }

    private double[][] readCSV(String filename) {
        List<Double> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();

        try (Reader reader = new FileReader(filename);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {

            for (CSVRecord record : csvParser) {
                xList.add(Double.parseDouble(record.get("X")));
                yList.add(Double.parseDouble(record.get("Y")));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] X = xList.stream().mapToDouble(Double::doubleValue).toArray();
        double[] Y = yList.stream().mapToDouble(Double::doubleValue).toArray();
        return new double[][]{X, Y};
    }
}
