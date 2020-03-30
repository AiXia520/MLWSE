package executeMulan.mlc;
import clus.algo.kNN.KNNClassifier;
import executeMulan.ExecuteMulanAlgorithm;
import mulan.classifier.transformation.MultiLabelStacking;
import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

public class ExecuteMLS extends ExecuteMulanAlgorithm {
	
	public void execute (String tvalue, String Tvalue, String xvalue, String ovalue, boolean lvalue)
	{		
		 try{
			 prepareExecution(tvalue, Tvalue, xvalue, ovalue);
			 
			 MultiLabelStacking learner = null;
            
		
            /* Only one execution (does not use random numbers) */
			 
        	time_in = System.currentTimeMillis();
        	   
        	learner = new MultiLabelStacking(new AdaBoostM1(),new J48());
    	    learner.build(trainingSet);
    	
    	    measures = prepareMeasuresClassification(trainingSet);    	       
    	    results = eval.evaluate(learner, testSet, measures);
    	       
    	    System.out.println("Multi-label Stacking:"+results);
    	    
    	    time_fin = System.currentTimeMillis();
    	      
    	    total_time = time_fin - time_in;

    	    System.out.println("Execution time (ms): " + total_time);

    	    printHeader(lvalue);
    	    printResults(Tvalue, lvalue, "MLS");
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
