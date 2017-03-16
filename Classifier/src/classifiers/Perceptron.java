package classifiers;

import preprocessor.Preprocessor.Mode;

/**
 * A Perceptron based classifier
 * @author Kenneth Chiguichon 109867025
 */
public class Perceptron {
    private int dimension;
    private double[] weightMatrix;
    private final Mode mode;
    public static enum Class{
        POSITIVE,
        NEGATIVE
    };
    
    public Perceptron(int dimension, Mode mode){
        this.dimension = dimension;
        weightMatrix = new double[dimension+1];
        this.mode = mode;
    }
    
    public void randomizeWeights(double bias){
        weightMatrix[0] = bias;
        for(int i = 1; i <= dimension; ++i) weightMatrix[i] = Math.random();
    }
    
    public void train(double[] vector, Class label){
        double output = 0;
        for(int i = 0; i <= dimension; ++i)
            output += vector[i] * weightMatrix[i];
    }
    
    public double squash(double input){
        return 1/(1+Math.exp(-input));
    }
    
    public Class test(double[] vector){
        return null;
    }
}
