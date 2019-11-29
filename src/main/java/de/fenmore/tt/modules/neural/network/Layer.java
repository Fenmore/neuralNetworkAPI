package de.fenmore.tt.modules.neural.network;

public class Layer {

    private double[] nodes;
    private Weights weightsIncoming;

    public Layer(int valueCount, Weights weightsIncoming) {
        nodes = new double[valueCount];
        this.weightsIncoming = weightsIncoming;
    }

    public int size() {
        return nodes.length;
    }

    public double get(int pos) {
        return nodes[pos];
    }

    public Weights getIncomingWeights() {
        return weightsIncoming;
    }

    public void set(int pos, double value) {
        nodes[pos] = value;
    }
}
