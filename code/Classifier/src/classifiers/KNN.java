package classifiers;

import classifiers.Perceptron.Label;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.BiFunction;
import preprocessor.Preprocessor.Mode;

public class KNN {
    public static enum Distance{
        EUCLIDEAN, MANHATTAN;
    };
    
    private Mode mode;
    private String[] dictionary;
    private BiFunction<Map<String,Integer>, Map<String, Integer>, Double> distance;
    private int K = 0;
    
    private HashMap<Map<String, Integer>, Label> vectorLabels;
    private HashMap<Map<String, Integer>, Double> vectorDistancePairs;
    private Map<String, Integer>[] heap;
    private int heapIndex = 0;
    
    private void init(int K, TreeSet<String> dictionary, Mode mode){
        this.K = K;
        vectorLabels = new HashMap<>();
        vectorDistancePairs = new HashMap<>();
        heap = (Map<String, Integer>[]) new Map<?,?>[K];
        this.mode = mode;
        this.dictionary = new String[dictionary.size()];
        this.dictionary = dictionary.toArray(this.dictionary);
    }
    
    public KNN(int K, TreeSet<String> dictionary, Mode mode, Distance distance){
        if(dictionary == null) throw new IllegalArgumentException();
        init(K, dictionary, mode);
        if(distance == Distance.MANHATTAN){
            this.distance = (p,q) -> {
                double result = 0, pNorm = 0, qNorm = 0;
                for(String s : p.keySet())
                    pNorm += Math.pow(p.get(s), 2);
                pNorm = Math.sqrt(pNorm);
                for(String s : q.keySet())
                    qNorm += Math.pow(q.get(s), 2);
                qNorm = Math.sqrt(qNorm);
                for(String s : p.keySet()){
                    if(q.containsKey(s)) 
                        result += Math.abs(
                            (p.get(s)/pNorm) - (q.get(s)/qNorm)
                        );
                    else result += p.get(s)/pNorm;
                }
                for(String s : q.keySet())
                    if(!p.containsKey(s))
                        result += q.get(s)/qNorm;
                return result;
            };
        }
        else {
            this.distance = (p,q) -> {
                double result = 0, pNorm = 0, qNorm = 0;
                for(String s : p.keySet())
                    pNorm += Math.pow(p.get(s), 2);
                pNorm = Math.sqrt(pNorm);
                for(String s : q.keySet())
                    qNorm += Math.pow(q.get(s), 2);
                qNorm = Math.sqrt(qNorm);
                for(String s : p.keySet()){
                    if(q.containsKey(s)) 
                        result += Math.pow(
                            (p.get(s)/pNorm) - (q.get(s)/qNorm), 2
                        );
                    else result += Math.pow(p.get(s)/pNorm,2);
                }
                for(String s : q.keySet())
                    if(!p.containsKey(s))
                        result += Math.pow(q.get(s)/qNorm,2);
                return Math.sqrt(result);
            };
        }
    }
    
    
    public KNN(int K, TreeSet<String> dictionary, Mode mode, BiFunction<Map<String,Integer>, Map<String, Integer>, Double> distance){
        if(dictionary == null) throw new IllegalArgumentException();
        init(K, dictionary, mode);
        this.distance  = distance;
    }
    
    private void fixHeap(int start, int end){
        int root = start, child, tempIndex;
        Map<String, Integer> temp;
        while((child = ((root << 1) + 1)) < end){
                tempIndex = root;
                if(vectorDistancePairs.get(heap[child]) > vectorDistancePairs.get(heap[tempIndex])) 
                    tempIndex = child;
                if(child + 1 < end && 
                    vectorDistancePairs.get(heap[child + 1]) > vectorDistancePairs.get(heap[tempIndex])) 
                    tempIndex = child + 1;
                if(tempIndex == root) break;
                temp = heap[tempIndex];
                heap[tempIndex] = heap[root];
                heap[root] = temp;
                root = tempIndex;
        }
    }
    
    private boolean addToHeap(Map<String, Integer> e){
        if(heapIndex >= K){
            // O((n-k) * log(k))
            if(vectorDistancePairs.get(e) < vectorDistancePairs.get(heap[0])){
                heap[0] = e;
                fixHeap(0, K);
                return true;
            } 
            else return false;
        }
        else{
            // O(k)
            int index = heapIndex++, parentIndex;
            Map<String, Integer> temp;
            heap[index] = e;
            while((parentIndex = ((index-1) >> 1)) >= 0 && 
                vectorDistancePairs.get(heap[index]) > vectorDistancePairs.get(heap[parentIndex])) {
                temp = heap[index];
                heap[index] = heap[parentIndex];
                heap[parentIndex] = temp;
            }
            return true;
        }
    }
    
    public void addVector(Map<String, Integer> e, Label label){
        vectorLabels.put(e, label);
    }
    
    public Label calculateLabel(Map<String, Integer> e){
        int numPos = 0, numNeg = 0;
        heapIndex = 0;
        vectorDistancePairs.clear();
        for(Map<String, Integer> vector : vectorLabels.keySet()){
            vectorDistancePairs.put(vector, distance.apply(e, vector));
            addToHeap(vector); 
        }
        for(Map<String, Integer> vector : heap){
            if(vectorLabels.get(vector) == Label.POSITIVE) ++numPos;
            else ++numNeg;
        }
        return (numPos >= numNeg)? Label.POSITIVE : Label.NEGATIVE;
    }
}
