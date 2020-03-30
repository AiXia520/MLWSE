package executeMulan.mlc;
import executeMulan.ExecuteMulanAlgorithm;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import weka.classifiers.trees.J48;

public class ExecuteRAkEL extends ExecuteMulanAlgorithm {
	
	public void execute (String tvalue, String Tvalue, String xvalue, String ovalue, boolean lvalue, int nIter)
	{		
		 try{
			 prepareExecution(tvalue, Tvalue, xvalue, ovalue);
			 
			 RAkEL learner = null;
            
			 /* The seeds are 10, 20, 30, ... */        	   
			 for(int i=1; i<=nIter; i++)
			 {
				time_in = System.currentTimeMillis();
				 
				learner = new RAkEL(new LabelPowerset(new J48()));
				learner.setSeed(i*10);

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
	    	    
	    	    printResults(Tvalue, lvalue, "RAkEL");
	    	    
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
