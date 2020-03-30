package executeMulan.mlc;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import calculateMLMetrics.LabelMatrix;
import executeMulan.ExecuteMulanAlgorithm;
import weka.core.Instances;
import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.Evaluation;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.*;


public class ExecuteRFPCT extends ExecuteMulanAlgorithm {
	
	public void execute(String tvalue, String Tvalue, String xvalue, String ovalue, boolean lvalue, int nIter)
	{		
		 try{
			 prepareExecution(tvalue, Tvalue, xvalue, ovalue);
			 
			 RFPCT learner = null;
            
			 /* The seeds are 10, 20, 30, ... */        	   
			 for(int i=1; i<=nIter; i++)
			 {
				time_in = System.currentTimeMillis();
				 
          	   	learner = new RFPCT("Clus.jar", 10, i*10);

          	   	learner.build(trainingSet);
    	       
	    	    measures = prepareMeasuresClassification(trainingSet);    	       
	    	    results = eval.evaluate(learner, testSet, measures);
	    	    
	    	    LabelMatrix lm = getLabelsClus(testSet.getNumInstances(), testSet.getNumLabels());
	    	    results = evaluate(lm.realLabels, lm.predLabels, measures);
	    	       
	    	    time_fin = System.currentTimeMillis();
	    	      
	    	    total_time = time_fin - time_in;
	
	    	    System.out.println("Execution time (ms): " + total_time);

	    	    //Print header only in first iteration
	    	    if(i == 1) {
	    	    	printHeader(lvalue);
	    	    }
	    	    
	    	    printResults(Tvalue, lvalue, "RF-PCT");
	    	    
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
	
	public static LabelMatrix getLabelsClus(int nInstances, int nLabels) throws FileNotFoundException, IOException{
		LabelMatrix lm = new LabelMatrix(nInstances, nLabels);
		
		Instances inst = new Instances(new FileReader("Clus.jardata-train.test.pred.arff"));

		System.out.println(inst.get(0).toString());
		for(int i=0; i<inst.numAttributes(); i++){
			System.out.print((int)inst.get(0).value(i) + " ");
		}
		System.out.println();
		
		for(int i=0; i<nInstances; i++){
			for(int l=0; l<nLabels; l++){
				int v = (int)inst.get(i).value(l);
				
				if(v == 1){
					lm.realLabels[i][l] = 0;
				}
				else{
					lm.realLabels[i][l] = 1;
				}
				
				v = (int)inst.get(i).value(l+nLabels);
				
				if(v == 1){
					lm.predLabels[i][l] = 0;
				}
				else{
					lm.predLabels[i][l] = 1;
				}
			}
		}
		
		return lm;
	}
	
	public static Evaluation evaluate(int [][] realLabels, int [][] predLabels,
	         List<Measure> measures) throws Exception {
//	        checkMeasures(measures);

	        // reset measures
	        for (Measure m : measures) {
	            m.reset();
	        }

	        int numLabels = realLabels[0].length;
	        Set<Measure> failed = new HashSet<Measure>();
//	        Instances testData = mlTestData.getDataSet();
	        int numInstances = realLabels.length;
	        for (int i = 0; i < numInstances; i++) {
//	            Instance instance = testData.instance(instanceIndex);
//	            boolean hasMissingLabels = mlTestData.hasMissingLabels(instance);
//	            Instance labelsMissing = (Instance) instance.copy();
//	            labelsMissing.setDataset(instance.dataset());
//	            for (int i = 0; i < mlTestData.getNumLabels(); i++) {
//	                labelsMissing.setMissing(labelIndices[i]);
//	            }
//	            MultiLabelOutput output = learner.makePrediction(labelsMissing);
	            
	            boolean [] predBool = new boolean[numLabels];
	            boolean [] realBool = new boolean[numLabels];
	            
	            for(int j=0; j<numLabels; j++){
	            	if(predLabels[i][j] == 1){
	            		predBool[j] = true;
	            	}
	            	else{
	            		predBool[j] = false;
	            	}
	            	
	            	if(realLabels[i][j] == 1){
	            		realBool[j] = true;
	            	}
	            	else{
	            		realBool[j] = false;
	            	}
	            }
	            
	            MultiLabelOutput output2 = new MultiLabelOutput(predBool);
	            GroundTruth truth2 = new GroundTruth(realBool);
	            
//	            if (output.hasPvalues()) {// check if we have regression outputs
//	                truth = new GroundTruth(getTrueScores(instance, numLabels, labelIndices));
//	            } else {
//	                truth = new GroundTruth(getTrueLabels(instance, numLabels, labelIndices));
//	            }
	            Iterator<Measure> it = measures.iterator();
	            while (it.hasNext()) {
	                Measure m = it.next();
	                if (!failed.contains(m)) {
	                    try {
	                        m.update(output2, truth2);
	                    } catch (Exception ex) {
	                        failed.add(m);
	                    }
	                }
	            }
	        }

	        return new Evaluation(measures, null);
	    }
	
}
