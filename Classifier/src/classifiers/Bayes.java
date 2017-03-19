package classifiers;

import classifiers.Perceptron.Label;
import java.io.File;
import java.util.TreeMap;
import java.util.TreeSet;
import preprocessor.Preprocessor.Mode;

/**
 * A naive Bayesian classifier
 * @author Kenneth Chiguichon 109867025
 */
public class Bayes {
    
    private final int dimension;
    private int numPos = 0, numNeg = 0;
    private String[] dictionary;
    private TreeMap<String, Integer> indexMap;
    private final Mode mode;
    private TreeMap<String, Integer> posCount, negCount;
    
    public Bayes(TreeSet<String> dictionary, int dimension, Mode mode){
        this.dimension = dimension;
        this.dictionary = new String[dimension];
        this.dictionary = dictionary.toArray(this.dictionary);
        this.indexMap = new TreeMap<>();
        for(int i = 0; i < dimension; ++i)
            indexMap.put(this.dictionary[i], i);
        this.mode = mode;
        this.posCount = new TreeMap<>();
        this.negCount = new TreeMap<>();
    }
    
    public void train(File file, TreeMap<String, Integer> vector, Label label){
        if(label == Label.POSITIVE){
            numPos++;
            vector.keySet().forEach((s) -> {
                posCount.put(s, (posCount.get(s) == null)? 1:posCount.get(s)+vector.get(s));
            });
        }
        else{
            numNeg++;
            vector.keySet().forEach((s) -> {
                negCount.put(s, (negCount.get(s) == null)? 1:negCount.get(s)+vector.get(s));
            });
        }
    }
    
    public Label test(TreeMap<String, Integer> vector){
        double posProb = 0, negProb = 0;
        final double m = 1., p = .5;
        for(String word : dictionary){
            if(vector.containsKey(word)){
                if(posCount.get(word) == null || posCount.get(word) == 0){
                    posProb += Math.log(m*p/(numPos+m));
                }
                else{
                    posProb += Math.log((double)posCount.get(word)/(double)numPos);
                }
                if(negCount.get(word) == null || negCount.get(word) == 0){
                    negProb += Math.log(m*p/(numNeg+m));
                }
                else{
                    negProb += Math.log((double)negCount.get(word)/(double)numNeg);
                }
            }
            else{
                double tempPos = (double)numPos - (double)((posCount.get(word) == null)?0:posCount.get(word)),
                        tempNeg = (double)numNeg - (double)((negCount.get(word) == null)?0:negCount.get(word));
                if(tempPos == 0){
                    posProb += Math.log(m*p/(numPos+m));
                }
                else{
                    posProb += Math.log(tempPos/(double)numPos);
                }
                if(tempNeg == 0){
                    negProb += Math.log(m*p/(numNeg+m));
                }
                else{
                    negProb += Math.log(tempNeg/(double)numNeg);
                }
            }
        }
        posProb += Math.log(((double)numPos)/((double)numPos+(double)numNeg));
        negProb += Math.log(((double)numNeg)/((double)numPos+(double)numNeg));
        return (posProb > negProb)? Label.POSITIVE : Label.NEGATIVE;
    }
}
