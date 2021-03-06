package classifiers;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import preprocessor.Preprocessor.Mode;

/**
 * A Perceptron based classifier
 * @author Kenneth Chiguichon 109867025
 */
public class Perceptron {
    private int dimension;
    private double[] weightMatrix;
    private String[] dictionary;
    private HashMap<String, Integer> indexMap;
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
    
    public Perceptron(TreeSet<String> dictionary, int dimension, double learningRate, Mode mode){
        this.dimension = dimension;
        weightMatrix = new double[dimension+1];
        this.dictionary = new String[dimension];
        this.dictionary = dictionary.toArray(this.dictionary);
        indexMap = new HashMap<>();
        for(int i = 0; i < dimension; ++i)
            indexMap.put(this.dictionary[i], i);
        this.learningRate = learningRate;
        this.mode = mode;
    }
    
    public Mode getMode() {
        return mode;
    }
         
    public void randomizeWeights(double bias){
        weightMatrix[0] = bias;
        Random random = new Random();
        for(int i = 1; i <= dimension; ++i) weightMatrix[i] = random.nextGaussian()*2 - 1;
    }
    
    public void train(Map<String, Integer> vector, Label label){
        double output = weightMatrix[0];
        int i = 1;
        if(mode == Mode.BINARY){
            for (String key : vector.keySet()) 
                output += vector.get(key) * weightMatrix[indexMap.get(key)+1];
        }
        else{
            for (String key : vector.keySet()) 
                output += (1+Math.log(vector.get(key))) * weightMatrix[indexMap.get(key)+1];
        }
        updateWeights(vector, squash(output), label);
    }
    
    private void updateWeights(Map<String, Integer> vector, Label guess, Label label){
        if(mode == Mode.BINARY)
            for (String key : vector.keySet()) 
                weightMatrix[indexMap.get(key)+1] = weightMatrix[indexMap.get(key)+1] + learningRate * (label.value - guess.value) * vector.get(key);
        else
            for (String key : vector.keySet()) 
                weightMatrix[indexMap.get(key)+1] = weightMatrix[indexMap.get(key)+1] + learningRate * (label.value - guess.value) * (1+Math.log(vector.get(key)));
    }
    
    private Label squash(double input){
        return (input < 0) ? Label.NEGATIVE : Label.POSITIVE;
    }
    
    public Label test(Map<String, Integer> vector){
        double output = weightMatrix[0];
        int i = 1;
        if(mode == Mode.BINARY){
            for (String key : vector.keySet()) 
                output += vector.get(key) * weightMatrix[indexMap.get(key)+1];
        }
        else{
            for (String key : vector.keySet()) 
                output += (1+Math.log(vector.get(key))) * weightMatrix[indexMap.get(key)+1];
        }
        return squash(output);
    }
}
