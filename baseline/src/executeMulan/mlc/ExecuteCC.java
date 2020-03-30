package executeMulan.mlc;
import java.util.Random;

import executeMulan.ExecuteMulanAlgorithm;
import mulan.classifier.transformation.ClassifierChain;
import weka.classifiers.trees.J48;

public class ExecuteCC extends ExecuteMulanAlgorithm {
	
	public void execute (String tvalue, String Tvalue, String xvalue, String ovalue, boolean lvalue, int nIter)
	{		
		 try{
			 prepareExecution(tvalue, Tvalue, xvalue, ovalue);
			 
			 ClassifierChain learner = null;
            
			 /* The seeds are 10, 20, 30, ... */        	   
			 for(int i=1; i<=nIter; i++)
			 {
				 time_in = System.currentTimeMillis();
				 Random rand = new Random(i*10);

          	   	//Get random chain
          	   	int [] chain = new int[trainingSet.getNumLabels()];
          	   	for(int c=0; c<trainingSet.getNumLabels(); c++) {
          	   		chain[c] = c;
          	   	}
          		   
          	   	for(int c=0; c<trainingSet.getNumLabels(); c++)
          	   	{
          	   		int r = rand.nextInt(trainingSet.getNumLabels());
          	   		int swap = chain[c];
          	   		chain[c] = chain[r];
          	   		chain[r] = swap;
          	   	}
          	  
          	   	learner = new ClassifierChain(new J48(), chain);

          	   	learner.build(trainingSet);
    	       
	    	    measures = prepareMeasuresClassification(trainingSet);    	       
	    	    results = eval.evaluate(learner, testSet, measures);
	    	       
	    	    time_fin = System.currentTimeMillis();
	    	      
	    	    total_time = time_fin - time_in;
	
	    	    System.out.println("Execution time (ms): " + total_time);

	    	    //Print header only in first iteration
	    	    if(i == 1) {
	    	    	printHeader(lvalue);
	    	    }
	    	    
	    	    printResults(Tvalue, lvalue, "CC");
	    	    
			 }//End for
			 
		}
        catch(Exception e1)
    	{
    		e1.printStackTrace();
    	}
    	finally{
    		if(pw != null)
    		{
    			pw.close();
    		}
    	}      
	}
}
