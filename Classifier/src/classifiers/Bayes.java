package classifiers;

import java.util.HashMap;
import java.util.TreeSet;
import preprocessor.Preprocessor.Mode;

/**
 *
 * @author Kenneth Chiguichon 109867025
 */
public class Bayes {
    
    private int dimension;
    private String[] dictionary;
    private HashMap<String, Integer> indexMap;
    private final Mode mode;
    
    public Bayes(TreeSet<String> dictionary, int dimension, Mode mode){
        this.dimension = dimension;
        this.dictionary = new String[dimension];
        this.dictionary = dictionary.toArray(this.dictionary);
        this.indexMap = new HashMap<>(dimension);
        for(int i = 0; i < dimension; ++i)
            indexMap.put(this.dictionary[i], i);
        this.mode = mode;
    }
    
    
}
