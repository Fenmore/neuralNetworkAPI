package de.fenmore.tt.modules.neural.network;

public class Weights {

    private double[][] weights;

    public Weights(int from, int to) {
        this.weights = new double[from][to];
        double value = 1.0 / from;
        for (int f = 0; f < from; f++) {
            for (int t = 0; t < to; t++) {
                weights[f][t] = value;
            }
        }
    }

    public double get(int a, int b) {
        return weights[a][b];
    }

    public void add(int a, int b, double v) {
        weights[a][b] += v;
    }
}
