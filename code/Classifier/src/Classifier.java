/**
 * Main Class
 * @author Kenneth Chiguichon 109867025
 */
import classifiers.Bayes;
import classifiers.KNN;
import classifiers.KNN.Distance;
import classifiers.Perceptron;
import classifiers.Perceptron.Label;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import preprocessor.Preprocessor;
import preprocessor.Preprocessor.Mode;
import static preprocessor.Preprocessor.PUNCTUATION;

public class Classifier {

    private static final int ITERATIONS = 60, FOLDS = 5;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        boolean perceptron = true;
        ArrayList<String> newArgs = new ArrayList<>();
        String[] newArgsArray;
        for(String arg : args){
            if(arg.trim().equals("--perceptron")){ 
                perceptron = true;
                continue;
            }
            else if(arg.trim().equals("--bayes")){
                perceptron = false;
                continue;
            }
            newArgs.add(arg);
        }
        
        newArgsArray = new String[newArgs.size()];
        newArgsArray = newArgs.toArray(newArgsArray);
        
        Preprocessor processor = new Preprocessor(newArgsArray);
        processor.partition(FOLDS);
        
        int testFold = 0;
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            processor.buildDictionary(i);
        }
        
        /*
        if(perceptron)
            trainAndTestPerceptron(processor, testFold);
        else 
            trainAndTestBayes(processor, testFold);
        */
        
        Classifier.trainAndTestKNN(processor, testFold);
        
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
                    TreeMap<String, Integer> vector = new TreeMap<>();
                    // Tokenize the file appropriately and transform into vector
                     try (
                            FileInputStream finStream = new FileInputStream(file); 
                            Scanner fileReader = new Scanner(finStream)
                        ) {
                            while(fileReader.hasNext()){
                                String next = fileReader.next();
                                if(processor.isPunctuation()){
                                    StringBuilder str = new StringBuilder();
                                    for(int j = 0; j < next.length(); ++j){
                                        int index = -1;
                                        if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                            if(str.length() > 0) {
                                                if(perceptron.getMode() == Mode.BINARY)
                                                    vector.put(str.toString(), 1);
                                                else 
                                                  vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                                str = new StringBuilder();
                                            }
                                            String temp = new StringBuilder().append(next.charAt(j)).toString();
                                            if(perceptron.getMode() == Mode.BINARY)
                                                vector.put(temp, 1);
                                            else 
                                              vector.put(temp, (vector.get(temp) == null)? 1 : vector.get(temp)+1);
                                        }
                                        else{
                                            str.append(next.charAt(j));
                                        }
                                    }
                                    if(str.length() > 0){ 
                                        if(perceptron.getMode() == Mode.BINARY)
                                            vector.put(str.toString(), 1);
                                        else 
                                          vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                    }
                                }
                                else {
                                    Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                          .forEach((s) ->{
                                              if(perceptron.getMode() == Mode.BINARY)
                                                  vector.put(s, 1);
                                              else 
                                                  vector.put(s, (vector.get(s) == null)? 1 : vector.get(s)+1);
                                          });
                                }
                            }
                            if(fileReader.ioException() != null) 
                                throw fileReader.ioException();
                        } catch (FileNotFoundException e) {
                            Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                        } catch (IOException e){
                            Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                        }
                     // feed vector to perceptron
                     perceptron.train(vector, processor.getFold(i).get(file));
                }
            }
        }
        
        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
                TreeMap<String, Integer> vector = new TreeMap<>();
                 try (
                        FileInputStream finStream = new FileInputStream(file); 
                        Scanner fileReader = new Scanner(finStream)
                    ) {
                        while(fileReader.hasNext()){
                            String next = fileReader.next();
                                if(processor.isPunctuation()){
                                    StringBuilder str = new StringBuilder();
                                    for(int j = 0; j < next.length(); ++j){
                                        int index = -1;
                                        if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                            if(str.length() > 0) {
                                                if(perceptron.getMode() == Mode.BINARY)
                                                    vector.put(str.toString(), 1);
                                                else 
                                                  vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                                str = new StringBuilder();
                                            }
                                            String temp = new StringBuilder().append(next.charAt(j)).toString();
                                            if(perceptron.getMode() == Mode.BINARY)
                                                vector.put(temp, 1);
                                            else 
                                              vector.put(temp, (vector.get(temp) == null)? 1 : vector.get(temp)+1);
                                        }
                                        else{
                                            str.append(next.charAt(j));
                                        }
                                    }
                                    if(str.length() > 0){ 
                                        if(perceptron.getMode() == Mode.BINARY)
                                            vector.put(str.toString(), 1);
                                        else 
                                          vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                    }
                                }
                                else {
                                    Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                          .forEach((s) ->{
                                              if(perceptron.getMode() == Mode.BINARY)
                                                  vector.put(s, 1);
                                              else 
                                                  vector.put(s, (vector.get(s) == null)? 1 : vector.get(s)+1);
                                          });
                                }
                        }
                        if(fileReader.ioException() != null) 
                            throw fileReader.ioException();
                    } catch (FileNotFoundException e) {
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    } catch (IOException e){
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    }
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
    
    public static void trainAndTestBayes(Preprocessor processor, int testFold){
        int dimension = processor.getDictionary().size();
        
        Bayes bayes = new Bayes(processor.getDictionary(), dimension, processor.getMode());
        
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            for(File file : processor.getFold(i).keySet()){
                TreeMap<String, Integer> vector = new TreeMap<>();
                 try (
                        FileInputStream finStream = new FileInputStream(file); 
                        Scanner fileReader = new Scanner(finStream)
                    ) {
                        while(fileReader.hasNext()){
                            String next = fileReader.next();
                            if(processor.isPunctuation()){
                                StringBuilder str = new StringBuilder();
                                for(int j = 0; j < next.length(); ++j){
                                    int index = -1;
                                    if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                        if(str.length() > 0) {
                                            if(processor.getMode() == Mode.BINARY)
                                                vector.put(str.toString(), 1);
                                            else 
                                              vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                            str = new StringBuilder();
                                        }
                                        String temp = new StringBuilder().append(next.charAt(j)).toString();
                                        if(processor.getMode() == Mode.BINARY)
                                            vector.put(temp, 1);
                                        else 
                                          vector.put(temp, (vector.get(temp) == null)? 1 : vector.get(temp)+1);
                                    }
                                    else{
                                        str.append(next.charAt(j));
                                    }
                                }
                                if(str.length() > 0){ 
                                    if(processor.getMode() == Mode.BINARY)
                                        vector.put(str.toString(), 1);
                                    else 
                                      vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                }
                            }
                            else {
                                Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                      .forEach((s) ->{
                                          if(processor.getMode() == Mode.BINARY)
                                              vector.put(s, 1);
                                          else 
                                              vector.put(s, (vector.get(s) == null)? 1 : vector.get(s)+1);
                                      });
                            }
                        }
                        if(fileReader.ioException() != null) 
                            throw fileReader.ioException();
                    } catch (FileNotFoundException e) {
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    } catch (IOException e){
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    }
                 bayes.train(file, vector, processor.getFold(i).get(file));
            }
        }

        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
                TreeMap<String, Integer> vector = new TreeMap<>();
                 try (
                        FileInputStream finStream = new FileInputStream(file); 
                        Scanner fileReader = new Scanner(finStream)
                    ) {
                        while(fileReader.hasNext()){
                            String next = fileReader.next();
                                if(processor.isPunctuation()){
                                    StringBuilder str = new StringBuilder();
                                    for(int j = 0; j < next.length(); ++j){
                                        int index = -1;
                                        if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                            if(str.length() > 0) {
                                                if(processor.getMode() == Mode.BINARY)
                                                    vector.put(str.toString(), 1);
                                                else 
                                                  vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                                str = new StringBuilder();
                                            }
                                            String temp = new StringBuilder().append(next.charAt(j)).toString();
                                            if(processor.getMode() == Mode.BINARY)
                                                vector.put(temp, 1);
                                            else 
                                              vector.put(temp, (vector.get(temp) == null)? 1 : vector.get(temp)+1);
                                        }
                                        else{
                                            str.append(next.charAt(j));
                                        }
                                    }
                                    if(str.length() > 0){ 
                                        if(processor.getMode() == Mode.BINARY)
                                            vector.put(str.toString(), 1);
                                        else 
                                          vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                    }
                                }
                                else {
                                    Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                          .forEach((s) ->{
                                              if(processor.getMode() == Mode.BINARY)
                                                  vector.put(s, 1);
                                              else 
                                                  vector.put(s, (vector.get(s) == null)? 1 : vector.get(s)+1);
                                          });
                                }
                        }
                        if(fileReader.ioException() != null) 
                            throw fileReader.ioException();
                    } catch (FileNotFoundException e) {
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    } catch (IOException e){
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    }
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
    
    
    public static void trainAndTestKNN(Preprocessor processor, int testFold){
        int dimension = processor.getDictionary().size();
        int K = 25; // 19, 25, 33
        KNN knn = new KNN(K, processor.getDictionary(), processor.getMode(), Distance.EUCLIDEAN);
        
        for(int i = 0; i < FOLDS; ++i){
            if(i == testFold) continue;
            for(File file : processor.getFold(i).keySet()){
                HashMap<String, Integer> vector = new HashMap<>();
                 try (
                        FileInputStream finStream = new FileInputStream(file); 
                        Scanner fileReader = new Scanner(finStream)
                    ) {
                        while(fileReader.hasNext()){
                            String next = fileReader.next();
                            if(processor.isPunctuation()){
                                StringBuilder str = new StringBuilder();
                                for(int j = 0; j < next.length(); ++j){
                                    int index = -1;
                                    if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                        if(str.length() > 0) {
                                            if(processor.getMode() == Mode.BINARY)
                                                vector.put(str.toString(), 1);
                                            else 
                                              vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                            str = new StringBuilder();
                                        }
                                        String temp = new StringBuilder().append(next.charAt(j)).toString();
                                        if(processor.getMode() == Mode.BINARY)
                                            vector.put(temp, 1);
                                        else 
                                          vector.put(temp, (vector.get(temp) == null)? 1 : vector.get(temp)+1);
                                    }
                                    else{
                                        str.append(next.charAt(j));
                                    }
                                }
                                if(str.length() > 0){ 
                                    if(processor.getMode() == Mode.BINARY)
                                        vector.put(str.toString(), 1);
                                    else 
                                      vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                }
                            }
                            else {
                                Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                      .forEach((s) ->{
                                          if(processor.getMode() == Mode.BINARY)
                                              vector.put(s, 1);
                                          else 
                                              vector.put(s, (vector.get(s) == null)? 1 : vector.get(s)+1);
                                      });
                            }
                        }
                        if(fileReader.ioException() != null) 
                            throw fileReader.ioException();
                    } catch (FileNotFoundException e) {
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    } catch (IOException e){
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    }
                 knn.addVector(vector, processor.getFold(i).get(file));
            }
        }

        int numTotal = 0, numCorrect = 0, truePos = 0, falsePos = 0, trueNeg = 0, falseNeg = 0;
        // Measure Accuracy on test fold
        for(File file : processor.getFold(testFold).keySet()){
                HashMap<String, Integer> vector = new HashMap<>();
                 try (
                        FileInputStream finStream = new FileInputStream(file); 
                        Scanner fileReader = new Scanner(finStream)
                    ) {
                        while(fileReader.hasNext()){
                            String next = fileReader.next();
                                if(processor.isPunctuation()){
                                    StringBuilder str = new StringBuilder();
                                    for(int j = 0; j < next.length(); ++j){
                                        int index = -1;
                                        if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                            if(str.length() > 0) {
                                                if(processor.getMode() == Mode.BINARY)
                                                    vector.put(str.toString(), 1);
                                                else 
                                                  vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                                str = new StringBuilder();
                                            }
                                            String temp = new StringBuilder().append(next.charAt(j)).toString();
                                            if(processor.getMode() == Mode.BINARY)
                                                vector.put(temp, 1);
                                            else 
                                              vector.put(temp, (vector.get(temp) == null)? 1 : vector.get(temp)+1);
                                        }
                                        else{
                                            str.append(next.charAt(j));
                                        }
                                    }
                                    if(str.length() > 0){ 
                                        if(processor.getMode() == Mode.BINARY)
                                            vector.put(str.toString(), 1);
                                        else 
                                          vector.put(str.toString(), (vector.get(str.toString()) == null)? 1 : vector.get(str.toString())+1);
                                    }
                                }
                                else {
                                    Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                          .forEach((s) ->{
                                              if(processor.getMode() == Mode.BINARY)
                                                  vector.put(s, 1);
                                              else 
                                                  vector.put(s, (vector.get(s) == null)? 1 : vector.get(s)+1);
                                          });
                                }
                        }
                        if(fileReader.ioException() != null) 
                            throw fileReader.ioException();
                    } catch (FileNotFoundException e) {
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    } catch (IOException e){
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    }
                 Label actual = processor.getFold(testFold).get(file), result = knn.calculateLabel(vector);
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
