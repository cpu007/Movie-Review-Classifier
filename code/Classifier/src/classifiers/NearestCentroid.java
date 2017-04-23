package classifiers;

import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;
import preprocessor.Preprocessor.Mode;

public class NearestCentroid {
     public static enum Distance{
        EUCLIDEAN, MANHATTAN;
    };
     
    private Mode mode;
    private String[] dictionary;
    private BiFunction<Map<String,Integer>, Map<String, Integer>, Double> distance;
    private TreeMap<String, Double> positiveCentroid, negativeCentroid;
    
    
}
