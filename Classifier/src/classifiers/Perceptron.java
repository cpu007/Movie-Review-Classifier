package classifiers;

import preprocessor.Preprocessor.Mode;

/**
 * A Perceptron based classifier
 * @author Kenneth Chiguichon 109867025
 */
public class Perceptron {
    private int dimension;
    private double[] weightMatrix;
    private double learningRate;
    private final Mode mode;
    public static enum Label{
        POSITIVE(1),
        NEGATIVE(-1);
        public final int value;
        private Label(int value){
            this.value = value;
        }
    };
    
    public Perceptron(int dimension, double learningRate, Mode mode){
        this.dimension = dimension;
        weightMatrix = new double[dimension+1];
        this.learningRate = learningRate;
        this.mode = mode;
    }
    
    public void randomizeWeights(double bias){
        weightMatrix[0] = bias;
        for(int i = 1; i <= dimension; ++i) weightMatrix[i] = Math.random();
    }
    
    public void train(double[] vector, Label label){
        double output = 0;
        for(int i = 0; i <= dimension; ++i)
            output += vector[i] * weightMatrix[i];
        updateWeights(vector, squash(output), label);
    }
    
    private void updateWeights(double[] vector, Label guess, Label label){
        for(int i = 1; i <= dimension; ++i){
            weightMatrix[i] = weightMatrix[i-1] + learningRate * (label.value - guess.value) * vector[i];
        }
    }
    
    public Label squash(double input){
        return (1/(1+Math.exp(-input)) < .5)? Label.NEGATIVE : Label.POSITIVE;
    }
    
    public Label test(double[] vector){
        return null;
    }
}
