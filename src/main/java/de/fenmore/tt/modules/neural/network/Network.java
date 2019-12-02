package de.fenmore.tt.modules.neural.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Represents the neural network. It works with basic back propagation and biases.
 * @author Tobias Thirolf
 */
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

    /**
     * Creates a new neural network.
     * @param nodeCounts The count of nodes for each layer. Position 0 represents the input layer, the
     *                   last position represents the output layer.
     * @param epsilon The learning value. It's commonly between 0.1 and 0.0001. Note that bigger values
     *                may lead to "overstep" the perfect value for a weight. Smaller values require more
     *                learn steps to get to the perfect weight value. The value should not be negative.
     * @param stdDeviation States whether the standard deviation of the network error should be calculated.
     *                     See {@link #learn(double[][])}.
     */
    public Network(int[] nodeCounts, double epsilon, boolean stdDeviation) {
        this.layers = new ArrayList<>();
        Weights weights = null;
        for (int i = 0; i < nodeCounts.length; i++) {

            if (i != 0) {
                weights = new Weights(nodeCounts[i - 1] + 1, nodeCounts[i]); // + 1 for the bias
            }

            this.layers.add(new Layer(nodeCounts[i] + 1, weights)); // + 1 for the bias
            this.layers.get(i).set(nodeCounts[i], 1); //Initializes the bias
        }
        this.epsilon = epsilon;
        this.stdDeviation = stdDeviation;
    }

    /**
     *
     * @return
     */
    public double getMaxError() {
        return maxError;
    }

    /**
     *
     * @return
     */
    public double getMinError() {
        return minError;
    }

    /**
     *
     * @return
     */
    public double getAverageError() {
        return averageError;
    }

    /**
     *
     * @return
     */
    public double getStdDeviationError() {
        return stdDeviationError;
    }

    /**
     *
     * @param stdDeviation
     */
    public void setStdDeviation(boolean stdDeviation) {
        this.stdDeviation = stdDeviation;
    }

    /**
     *
     * @param epsilon
     */
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

    /**
     * Trains the network with given examples. Values for {@link #getAverageError()}, {@link #getMaxError()},
     * {@link #getMinError()}, {@link #getStdDeviationError()} will be updated here. The preparations of the
     * standard deviation are calculated after each example, the standard deviation itself at the end of this
     * method.
     * @param examples Each double[] within "examples" represents an example containing the input information
     *                 and expected outputs. Outputs have to be placed after the inputs. The length of input
     *                 and output information have to be the same as specified in
     *                 {@link #Network(int[], double, boolean)}.
     */
    public void learn(double[][] examples) {

        maxError = Double.MIN_VALUE;
        minError = Double.MAX_VALUE;
        averageError = 0.0;

        double summXQ = 0.0;
        double summXi = 0.0;

        int len = examples.length;
        for (double[] anExample : examples) {

            double[] d = new double[layers.get(layers.size() - 1).size() - 1]; // - 1 to exclude the bias

            for (int i = 0; i < layers.get(0).size() - 1; i++) { // - 1 to exclude the bias
                layers.get(0).set(i, anExample[i]);
            }
            for (int i = 0; i < layers.get(layers.size() - 1).size() - 1; i++) { // - 1 to exclude the bias
                d[i] = anExample[layers.get(0).size() - 1 + i]; // - 1 cause the bias is not part of the examples
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

    /**
     * Executes the current network stance to get the output for a given input.
     * @param input The values given as input. Giving less values than specified in
     *              {@link #Network(int[], double, boolean)} for the input layer will lead to a
     *              {@link IndexOutOfBoundsException}. Values which exceed the specified input
     *              layer node count will be ignored.
     * @return The values of the output layer after executing the network. The value count equals
     *          the specified count in {@link #Network(int[], double, boolean)} for the output layer.
     */
    public double[] execute(double[] input) {

        for (int i = 0; i < layers.get(0).size() - 1; i++) {
            layers.get(0).set(i, input[i]);
        }

        execute();

        Layer outputLayer = layers.get(layers.size() - 1);
        double[] output = new double[outputLayer.size() - 1]; // -1 to exclude the bias
        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.get(i);
        }

        return output;
    }
}