package preprocessor;

import classifiers.Perceptron.Label;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Preprocesses the data
 * @author Kenneth Chiguichon 109867025
 */
public class Preprocessor {
    
    private static int FOLDS = 5;
    private String posDirectory, negDirectory;
    private ArrayList<TreeMap<File, Label>> folds;
    private TreeSet<String> dictionary;
    public static final String PUNCTUATION = "!.,\\/",
                                NUMBER = "0123456789";
    public static enum Mode{
        FREQUENCY,
        BINARY
    };
    
    private Mode mode;
    private boolean punctuation;

    public boolean isPunctuation() {
        return punctuation;
    }
    
    public Preprocessor() {}
    
    public Preprocessor(String args[]){
        processArgs(args);
    }
    
    private static String[] help_menu = { 
        "A preprocessor tool to be used to aggregate and ready data for "+
        "classification\n" ,
        "Usage: java Preprocessor <flags> posDirectory negDirectory <flags>\n",
        "Flags:",
            "\t--help\t\tPrints the help menu",
            "\t--frequency\t\tAggregate reviews based on word frequency",
            "\t--binary\t\tAggregate reviews based on word count",
            "\t--nopunct\t\tDisregard punctuation",
            "\t--punct\t\tEnforce punctuation"
    };
    
    public static void main(String args[]){
        Preprocessor processor = new Preprocessor();
        processor.processArgs(args);
        processor.partition(FOLDS);
        processor.buildDictionary(0);
        System.out.println("Tokens parsed:");
        processor.printDictionary();
    }
    
    public static void print_help(boolean exit, int status){
        // Print the help menu
        Arrays.stream(help_menu).forEach(System.out::println);
        // Exit if necessary with appropriate status
        if(exit) System.exit(status);
    }
    
    public void processArgs(String[] args) throws IllegalArgumentException {
        for(String arg : args){
            if(arg.trim().startsWith("--")){
                String flag = arg.trim().substring(2).toLowerCase();
                switch (flag) {
                    case "help":
                        print_help(true, 0);
                        break;
                    case "frequency":
                        mode = Mode.FREQUENCY;
                        break;
                    case "binary":
                        mode = Mode.BINARY;
                        break;
                    case "nopunct":
                        punctuation = false;
                        break;
                    case "punct":
                        punctuation = true;
                        break;
                    default:
                        print_help(true, -1);
                        break;
                }
            }
            else{
                if(posDirectory == null){
                    posDirectory = arg;
                }
                else{
                    negDirectory = arg;
                }
            }
        }
        if (posDirectory == null || negDirectory == null) 
            throw new IllegalArgumentException();
    }
    
    public void partition(int numFolds) {
        folds = new ArrayList<>(numFolds);
        for(int i = 0; i < numFolds; ++i) folds.add(new TreeMap<>());
        File[]  posFiles = new File(posDirectory).listFiles(),
                negFiles = new File(negDirectory).listFiles();
        
        // Build partitions of data (k-folds)
        for(int i = 0, j = 0, k = 0; i < posFiles.length || j < negFiles.length; ++i, ++j, ++k){
            if(i < posFiles.length) folds.get(k % numFolds).put(posFiles[i], Label.POSITIVE);
            if(j < negFiles.length) folds.get(k % numFolds).put(negFiles[j], Label.NEGATIVE);
        }
    }
    
    public void buildDictionary(int testFold){
        if(testFold >= folds.size()) return;
        if(dictionary == null) dictionary = new TreeSet<>();
        for(int i = 0; i < folds.size(); ++i){
            if(i != testFold) {
                for(File file : folds.get(i).keySet()){
                    try (
                        FileInputStream finStream = new FileInputStream(file); 
                        Scanner fileReader = new Scanner(finStream)
                    ) {
                        while(fileReader.hasNext()){
                            String next = fileReader.next();
                            if(punctuation){
                                StringBuilder str = new StringBuilder();
                                for(int j = 0; j < next.length(); ++j){
                                    int index = -1;
                                    if((index = PUNCTUATION.indexOf(next.charAt(j))) > 0){
                                        if(str.length() > 0) {
                                            dictionary.add(str.toString());
                                            str = new StringBuilder();
                                        }
                                        dictionary.add(new StringBuilder().append(next.charAt(j)).toString());
                                    }
                                    else{
                                        str.append(next.charAt(j));
                                    }
                                }
                                if(str.length() > 0) dictionary.add(str.toString());
                            }
                            else {
                                Arrays.stream(next.split("["+PUNCTUATION+"]"))
                                      .forEach(dictionary::add);
                            }
                        }
                        if(fileReader.ioException() != null) 
                            throw fileReader.ioException();
                    } catch (FileNotFoundException e) {
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    } catch (IOException e){
                        Logger.getLogger(Preprocessor.class.getName()).log(Level.SEVERE, null, e);
                    }
                }
            }
        }
    }
    
    public TreeSet<String> getDictionary(){
        return dictionary;
    }
    
    public TreeMap<File, Label> getFold(int index){
        return folds.get(index);
    }
    
    public void printDictionary(){
        dictionary.forEach(System.out::println);
    }
}
