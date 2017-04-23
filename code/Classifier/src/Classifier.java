/**
 * Main Class
 * @author Kenneth Chiguichon 109867025
 */
import classifiers.Bayes;
import classifiers.Distance;
import classifiers.KNN;
import classifiers.NearestCentroid;
import classifiers.Perceptron;
import classifiers.Perceptron.Label;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import preprocessor.Preprocessor;

public class Classifier {
    
    public static enum CLASSIFIERS{
        BAYES, KNN, NEAREST_CENTROID, PERCEPTRON
    };
    
    private static final int ITERATIONS = 60, FOLDS = 5;
    
    private final static String[] HELP_MENU = { 
        "A movie review classification tool which supports multiple methods of "
        + "classification.\n" ,
        "Usage: java Classifier <flags> posDirectory negDirectory <flags>\n",
        "Flags:",
            "\t--help\t\t\tPrints the help menu",
            "\t--perceptron\t\tUses a perceptron for classification",
            "\t--bayes\t\t\tUses a naive Bayes classifier for classification",
            "\t--knn\t\t\tUses a K-Nearest-Neighbors classifier for classification",
            "\t--nearest-centroid\tUses a Rocchio classifier for classification",
            "\t--frequency\t\tAggregate reviews based on word frequency",
            "\t--binary\t\tAggregate reviews based on word count",
            "\t--nopunct\t\tDisregard punctuation",
            "\t--punct\t\t\tEnforce punctuation",
            "\t--k=x\t\t\tSets the K parameter in K nearest neighbors to x. (x > 0)",
            "\t--metric=X\t\tSets the distance metric as x for KNN and Rocchio classifiers. (x = {euclidean, manhattan})"
    };
    
    public static void main(String[] args) {
        CLASSIFIERS classifier = CLASSIFIERS.PERCEPTRON;
        ArrayList<String> newArgs = new ArrayList<>();
        String[] newArgsArray;
        int K = 13; // 5, 9, 13, 15
        Distance distance = Distance.EUCLIDEAN;
        for(String arg : args){
            if(arg.trim().equalsIgnoreCase("--perceptron"))
                classifier = CLASSIFIERS.PERCEPTRON;
            else if(arg.trim().equalsIgnoreCase("--bayes"))
                classifier = CLASSIFIERS.BAYES;
            else if(arg.trim().equalsIgnoreCase("--nearest-centroid"))
                classifier = CLASSIFIERS.NEAREST_CENTROID;
            else if(arg.trim().equalsIgnoreCase("--knn"))
                classifier = CLASSIFIERS.KNN;
            else if(arg.trim().toLowerCase().matches("-{2}k=\\d+"))
                K = Integer.parseInt(arg.trim().substring(4));
            else if(arg.trim().matches("-{2}metric=[a-zA-Z]+")){
                for(Distance d : Distance.values()){
                    if(arg.trim().substring(9).equalsIgnoreCase(d.toString())){
                        distance = d;
                        break;
                    }
                }
            }
            else if(arg.trim().equalsIgnoreCase("--help"))
                printHelp();
            else 
                newArgs.add(arg);
        }
        
        newArgsArray = new String[newArgs.size()];
        newArgsArray = newArgs.toArray(newArgsArray);
        
        Preprocessor processor = new Preprocessor(newArgsArray);
        processor.partition(FOLDS);
        
        int testFold = 3;
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            processor.buildDictionary(i);
        }
        
        switch(classifier){
            case PERCEPTRON: trainAndTestPerceptron(processor, testFold); break;
            case BAYES: trainAndTestBayes(processor, testFold); break;
            case KNN: trainAndTestKNN(processor, testFold, K, distance); break;
            case NEAREST_CENTROID: trainAndTestNearestCentroid(processor, testFold, distance); break;
        }
        
    }
    
    private static void printHelp() {
        Arrays.stream(HELP_MENU).forEach(System.out::println);
        System.exit(0);
    }
    
    public static void trainAndTestPerceptron(Preprocessor processor, int testFold){
        int dimension = processor.getDictionary().size();
        
        Perceptron perceptron = new Perceptron(processor.getDictionary(), dimension, 1, processor.getMode());
        perceptron.randomizeWeights(0);
        
        for(int k = 0; k < ITERATIONS; ++k){
            // Feed training folds to perceptron
            for(int i = 0; i < FOLDS; ++i){
                if(i == testFold) continue;
                // Process Files in each training fold
                for(File file : processor.getFold(i).keySet()){
                    Map<String, Integer> vector = processor.getVectorFromFile(file);
                    // feed vector to perceptron
                    perceptron.train(vector, processor.getFold(i).get(file));
                }
            }
        }
        
        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
            Map<String, Integer> vector = processor.getVectorFromFile(file);
             Label actual = processor.getFold(testFold).get(file), result = perceptron.test(vector);
             if(result == Label.POSITIVE) {
                 if(result == actual) truePos++;
                 else falsePos++;
             }
             else {
                 if(result == actual) trueNeg++;
                 else falseNeg++;
             }
             if(result == actual) numCorrect++;
             numTotal++;
        }
        printStatistics(numTotal, numCorrect, truePos, falsePos, trueNeg, falseNeg);
    }
    
    public static void trainAndTestBayes(Preprocessor processor, int testFold){
        int dimension = processor.getDictionary().size();
        
        Bayes bayes = new Bayes(processor.getDictionary(), dimension, processor.getMode());
        
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            for(File file : processor.getFold(i).keySet()){
                Map<String, Integer> vector = processor.getVectorFromFile(file);
                bayes.train(file, vector, processor.getFold(i).get(file));
            }
        }

        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
            Map<String, Integer> vector = processor.getVectorFromFile(file);
            Label actual = processor.getFold(testFold).get(file), result = bayes.test(vector);
            if(result == Label.POSITIVE) {
                if(result == actual) truePos++;
                else falsePos++;
            }
            else {
                if(result == actual) trueNeg++;
                else falseNeg++;
            }
            if(result == actual) numCorrect++;
            numTotal++;
        }
        printStatistics(numTotal, numCorrect, truePos, falsePos, trueNeg, falseNeg);
    }
    
    public static void trainAndTestKNN(Preprocessor processor, int testFold, int K, Distance distance){
        KNN knn = new KNN(K, distance);
        
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            for(File file : processor.getFold(i).keySet()){
                Map<String, Integer> vector = processor.getVectorFromFile(file);
                knn.addVector(vector, processor.getFold(i).get(file));
            }
        }

        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
            Map<String, Integer> vector = processor.getVectorFromFile(file);
            Label actual = processor.getFold(testFold).get(file), result = knn.testVector(vector);
            if(result == Label.POSITIVE) {
                if(result == actual) truePos++;
                else falsePos++;
            }
            else {
                if(result == actual) trueNeg++;
                else falseNeg++;
            }
            if(result == actual) numCorrect++;
            numTotal++;
        }
        printStatistics(numTotal, numCorrect, truePos, falsePos, trueNeg, falseNeg);
    }
    
    public static void trainAndTestNearestCentroid(Preprocessor processor, int testFold, Distance distance){
        NearestCentroid nearestCentroid = new NearestCentroid(processor.getDictionary(), distance);
        
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            for(File file : processor.getFold(i).keySet()){
                Map<String, Integer> vector = processor.getVectorFromFile(file);
                nearestCentroid.addVector(vector, processor.getFold(i).get(file));
            }
        }
        
        nearestCentroid.finishedTraining();

        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
            Map<String, Integer> vector = processor.getVectorFromFile(file);
            Label actual = processor.getFold(testFold).get(file), result = nearestCentroid.testVector(vector);
            if(result == Label.POSITIVE) {
                if(result == actual) truePos++;
                else falsePos++;
            }
            else {
                if(result == actual) trueNeg++;
                else falseNeg++;
            }
            if(result == actual) numCorrect++;
            numTotal++;
        }
        printStatistics(numTotal, numCorrect, truePos, falsePos, trueNeg, falseNeg);
    }
    
    private static void printStatistics(double numTotal, double numCorrect, 
            double truePos, double falsePos, double trueNeg, double falseNeg){
        double precisionPos = 0, precisionNeg = 0, recallPos = 0, recallNeg = 0;
        System.out.println("Test Data:");
        System.out.println("\nTotal: "+numTotal+"\nCorrect = "+numCorrect);
        System.out.println("True Positives = "+truePos);
        System.out.println("False Positives = "+falsePos);
        System.out.println("True Negatives = "+trueNeg);
        System.out.println("False Negatives = "+falseNeg);
        System.out.println("Precision+ = "+(precisionPos = (((double)truePos)/((double)truePos+(double)falsePos))));
        System.out.println("Precision- = "+(precisionNeg = (((double)trueNeg)/((double)trueNeg+(double)falseNeg))));
        System.out.println("Recall+ = "+(recallPos = (((double)truePos)/((double)truePos+(double)falseNeg))));
        System.out.println("Recall- = "+(recallNeg = (((double)trueNeg)/((double)trueNeg+(double)falsePos))));
        System.out.println("Accuracy = "+((double)numCorrect/(double)numTotal)*100.+ "%");
        System.out.println("Precision = "+(precisionPos+precisionNeg)/2.);
        System.out.println("Recall = "+(recallPos+recallNeg)/2.);
    }
}
