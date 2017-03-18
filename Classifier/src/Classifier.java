/**
 * Main Class
 * @author Kenneth Chiguichon 109867025
 */
import classifiers.Perceptron;
import classifiers.Perceptron.Label;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import preprocessor.Preprocessor;
import preprocessor.Preprocessor.Mode;
import static preprocessor.Preprocessor.PUNCTUATION;

public class Classifier {

    private static final int ITERATIONS = 50, FOLDS = 5;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Preprocessor processor = new Preprocessor(args);
        processor.partition(FOLDS);
        for(int i = 0; i < FOLDS-1; ++i){
            processor.buildDictionary(i);
        }
        int dimension = processor.getDictionary().size();
        
        Perceptron perceptron = new Perceptron(processor.getDictionary(), dimension, 1, processor.getMode());
        perceptron.randomizeWeights(0);
        
        for(int k = 0; k < ITERATIONS; ++k){
            for(int i = 0; i < FOLDS-1; ++i){
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
                     perceptron.train(vector, processor.getFold(i).get(file));
                }
            }
        }
        int numTotal = 0, numCorrect = 0;
        for(int i = 0; i < FOLDS-1; ++i){
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
                 Label correct = processor.getFold(i).get(file);
                 
                 if(perceptron.test(vector) == correct) numCorrect++;
                 numTotal++;
            }
        }
        System.out.println("Training Data:");
        System.out.println("Total: "+numTotal+"\nCorrect = "+numCorrect);
        System.out.println("Accuracy = "+((double)numCorrect/(double)numTotal)*100.+ "%");
        numTotal = 0;
        numCorrect = 0;
        for(File file : processor.getFold(FOLDS-1).keySet()){
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
                 Label correct = processor.getFold(FOLDS-1).get(file);
                 
                 if(perceptron.test(vector) == correct) numCorrect++;
                 numTotal++;
            }
        System.out.println("Test Data:");
        System.out.println("\nTotal: "+numTotal+"\nCorrect = "+numCorrect);
        System.out.println("Accuracy = "+((double)numCorrect/(double)numTotal)*100.+ "%");
    }
    
}
