package classifiers;

import classifiers.Perceptron.Label;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.BiFunction;

public class NearestCentroid {
     
    private String[] dictionary;
    private final BiFunction<Map<String,Integer>, Map<String, Double>, Double> distance;
    private HashMap<String, Double> positiveCentroid, negativeCentroid;
    
    private void init(TreeSet<String> dictionary){
        this.dictionary = new String[dictionary.size()];
        this.dictionary = dictionary.toArray(this.dictionary);
        positiveCentroid = new HashMap<>();
        negativeCentroid = new HashMap<>();
        dictionary.forEach((s) -> {
            positiveCentroid.put(s, 0d);
            negativeCentroid.put(s, 0d);
        });
    }
    
    public NearestCentroid(TreeSet<String> dictionary, Distance distance){
        init(dictionary);
        if(distance == Distance.MANHATTAN){
            this.distance = (p,q) -> {
                double result = 0, qNorm = 0;
                for(String s : q.keySet())
                    qNorm += Math.pow(q.get(s), 2);
                qNorm = Math.sqrt(qNorm);
                for(String s : p.keySet()){
                    if(q.containsKey(s)) 
                        result += Math.abs(
                            (p.get(s)) - (q.get(s)/qNorm)
                        );
                    else result += p.get(s);
                }
                for(String s : q.keySet())
                    if(!p.containsKey(s))
                        result += q.get(s)/qNorm;
                return result;
            };
        }
        else {
            this.distance = (p,q) -> {
                double result = 0, qNorm = 0;
                for(String s : q.keySet())
                    qNorm += Math.pow(q.get(s), 2);
                qNorm = Math.sqrt(qNorm);
                for(String s : p.keySet()){
                    if(q.containsKey(s)) 
                        result += Math.pow(
                            (p.get(s)) - (q.get(s)/qNorm), 2
                        );
                    else result += Math.pow(p.get(s),2);
                }
                for(String s : q.keySet())
                    if(!p.containsKey(s))
                        result += Math.pow(q.get(s)/qNorm,2);
                return Math.sqrt(result);
            };
        }
    }
    
    public NearestCentroid(TreeSet<String> dictionary, 
        BiFunction<Map<String,Integer>, Map<String, Double>, Double> distance){
        init(dictionary);
        this.distance = distance;
    }
    
    public void finishedTraining(){
        double pNorm = 0, nNorm = 0, temp;
        for(String s : dictionary){
            positiveCentroid.put(s, (temp = (positiveCentroid.get(s)/positiveCentroid.size())));
            pNorm += temp * temp;
            negativeCentroid.put(s, (temp = (negativeCentroid.get(s)/negativeCentroid.size())));
            nNorm += temp * temp;
        }
        pNorm = Math.sqrt(pNorm);
        nNorm = Math.sqrt(nNorm);
        for(String s : dictionary){
            positiveCentroid.put(s, positiveCentroid.get(s)/pNorm);
            negativeCentroid.put(s, negativeCentroid.get(s)/nNorm);
        }
    }
    
    public void addVector(Map<String, Integer> vector, Label label){
        if(label == Label.POSITIVE)
            vector.forEach((x,y) -> {positiveCentroid.put(x, positiveCentroid.get(x)+y);});
        else 
            vector.forEach((x,y) -> {negativeCentroid.put(x, negativeCentroid.get(x)+y);});
    }
    
    public Label testVector(Map<String, Integer> vector){
        return (distance.apply(vector, positiveCentroid) < distance.apply(vector, negativeCentroid))? Label.POSITIVE : Label.NEGATIVE;
    }
}
