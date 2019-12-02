package de.fenmore.tt.modules.neural.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network {

    private List<Layer> layers;
    private double maxError;
    private double minError;
    private double averageError;
    private double maxErrorNet;
    private double minErrorNet;
    private boolean stdDeviation;
    private double stdDeviationError;
    private double epsilon;

    public Network(int[] nodeCounts, double epsilon, boolean stdDeviation) {
        this.layers = new ArrayList<>();
        Weights weights = null;
        for (int i = 0; i < nodeCounts.length; i++) {

            if (i != 0) {
                weights = new Weights(nodeCounts[i - 1] + 1, nodeCounts[i]); // + 1 for the bias
            }

            this.layers.add(new Layer(nodeCounts[i] + 1, weights)); // + 1 for the bias
        }
        this.epsilon = epsilon;
        this.stdDeviation = stdDeviation;
    }

    public double getMaxError() {
        return maxError;
    }

    public double getMinError() {
        return minError;
    }

    public double getAverageError() {
        return averageError;
    }

    public double getStdDeviationError() {
        return stdDeviationError;
    }

    public void setStdDeviation(boolean stdDeviation) {
        this.stdDeviation = stdDeviation;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    private void execute() {

        for (int l = 1; l < layers.size(); l++) {
            Layer lastLayer = layers.get(l - 1);
            Layer nextLayer = layers.get(l);

            for (int h = 0; h < nextLayer.size() - 1; h++) { // - 1 to exclude the bias
                double a = 0.0;
                for (int i = 0; i < lastLayer.size(); i++) {
                    a += (nextLayer.getIncomingWeights().get(i, h) * lastLayer.get(i));
                }
                nextLayer.set(h, transfer(a));
            }
        }

    }

    private double transfer(double a) {
        return (1.0 / (1.0 + Math.exp(-a)));
    }

    public void learning(double[][] example) {

        maxError = Double.MIN_VALUE;
        minError = Double.MAX_VALUE;
        averageError = 0.0;

        double summXQ = 0.0;
        double summXi = 0.0;

        int len = example.length;
        for (double[] anExample : example) {

            double[] d = new double[layers.get(layers.size() - 1).size() - 1]; // - 1 to exclude the bias

            for (int i = 0; i < layers.get(0).size() - 1; i++) { // - 1 to exclude the bias
                layers.get(0).set(i, anExample[i]);
            }
            for (int i = 0; i < layers.get(layers.size() - 1).size() - 1; i++) { // - 1 to exclude the bias
                d[i] = anExample[layers.get(0).size() - 1 + i]; // - 1 cause the bias is not included in the examples
            }

            execute();
            backPropagate(d);

            if (maxErrorNet > maxError) maxError = maxErrorNet;
            if (minErrorNet < minError) minError = minErrorNet;

            averageError += maxErrorNet;
            if (stdDeviation) {
                summXQ += Math.pow(maxErrorNet, 2);
                summXi += maxErrorNet;
            }
        }

        double n = len;
        averageError = averageError / n;
        if (stdDeviation) {
            double variance = (1 / (n - 1)) * (summXQ - (1 / n * Math.pow(summXi, 2)));
            stdDeviationError = Math.sqrt(variance);
        }
    }

    private void backPropagate(double[] d) {

        double[] lastHiddenError;
        double[] nextHiddenError = new double[0];
        maxErrorNet = Double.MIN_VALUE;
        minErrorNet = Double.MAX_VALUE;

        for (int l = layers.size() - 1; l >= 1; l--) {

            Layer layer = layers.get(l);
            Layer nextLayer = layers.get(l - 1);
            lastHiddenError = Arrays.copyOf(nextHiddenError, nextHiddenError.length);
            nextHiddenError = new double[nextLayer.size()];
            Weights weights = layer.getIncomingWeights();
            for (int j = 0; j < layer.size() - 1; j++) {

                double delta;
                if (l == layers.size() - 1) {
                    delta = Math.abs(d[j] - layer.get(j));
                    if (delta > maxErrorNet) maxErrorNet = delta;
                    if (delta < minErrorNet) minErrorNet = delta;

                    delta = (d[j] - layer.get(j)) * layer.get(j) * (1.0 - layer.get(j));
                }
                else {
                    delta = lastHiddenError[j] * layer.get(j) * (1.0 - layer.get(j));
                }

                for (int k = 0; k < nextLayer.size(); k++) {
                    nextHiddenError[k] += delta * weights.get(k, j);
                    weights.add(k, j, epsilon * delta * nextLayer.get(k));
                }
            }
        }
    }
}