package executeMulan.mlc;
import executeMulan.ExecuteMulanAlgorithm;
import mulan.classifier.transformation.BinaryRelevance;
import weka.classifiers.trees.J48;

public class ExecuteBR extends ExecuteMulanAlgorithm {
	
	public void execute (String tvalue, String Tvalue, String xvalue, String ovalue, boolean lvalue)
	{		
		 try{
			 prepareExecution(tvalue, Tvalue, xvalue, ovalue);
			 
			 BinaryRelevance learner = null;
            
            /* Only one execution (does not use random numbers) */
			
        	time_in = System.currentTimeMillis();
        	   
        	learner = new BinaryRelevance(new J48());
    	    learner.build(trainingSet);
    	       
    	    measures = prepareMeasuresClassification(trainingSet);    	       
    	    results = eval.evaluate(learner, testSet, measures);
    	       
    	    time_fin = System.currentTimeMillis();
    	      
    	    total_time = time_fin - time_in;

    	    System.out.println("Execution time (ms): " + total_time);

    	    printHeader(lvalue);
    	    printResults(Tvalue, lvalue, "BR");
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
