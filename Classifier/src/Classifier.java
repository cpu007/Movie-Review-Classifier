/**
 * Main Class
 * @author Kenneth Chiguichon 109867025
 */
import preprocessor.Preprocessor;

public class Classifier {

    private static final int FOLDS = 5;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Preprocessor processor = new Preprocessor(args);
        processor.partition(FOLDS);
        for(int i = 0; i < FOLDS; ++i){
            processor.buildDictionary(i);
            // TODO: train on ith fold
        }
        
    }
    
}
