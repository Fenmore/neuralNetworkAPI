package de.fenmore.tt.modules.neural.network;

public class Layer {

    private double[] nodes;
    private Weights weightsIncoming;

    public Layer(int nodeCount, Weights weightsIncoming) {
        nodes = new double[nodeCount];
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

    public void set(int pos, double nodeValue) {
        nodes[pos] = nodeValue;
    }
}
