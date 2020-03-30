package com.baseline.ensemble;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import clus.algo.kNN.KNNClassifier;
import executeMulan.mlc.EBR;
import executeMulan.mlc.ECC;
import executeMulan.mlc.EPS;
import executeMulan.mlc.ExecuteEBR;
import executeMulan.mlc.RFPCT;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.EnsembleOfSubsetLearners;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.AdaBoostMH;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.MultiLabelStacking;
import mulan.data.ConditionalDependenceIdentifier;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Utils;
import weka.core.pmml.jaxbbindings.NeuralNetwork;

public class Main {
	
	/**
	 * seven state-of-the-art ensemble multi-label classification methods 
	 * including EBR, ECC , EPS, RAkEL , CDE , AdaBoost.MH and MLS.
	 */
	public static void main(String[] args) throws Exception {
			

		  //Load datasets
		  List<String> list_dataset= new ArrayList<String>(Arrays.asList("yeast","enron", "3sources_bbc1000" ,"scene","CHD_49","Langlog")); 
		  	for (String ds:list_dataset) {
		  		
			  	String data_arff="src/data/"+ds+".arff";
				String data_xml="src/data/"+ds+".xml";
				
				System.out.println("********************"+ds+"datasets*********************");
		
		        MultiLabelInstances dataset = new MultiLabelInstances(data_arff, data_xml);
		        String method="";
		        Evaluator eval = new Evaluator();  
		        MultipleEvaluation results;
		        int numFolds = 5;
		        //Load methods
		        List<String> list= new ArrayList<String>(Arrays.asList("EBR", "ECC", "EPS",
		        		"RAkEL","AdaBoostMH","CDE","MLS")); 
		        for (String s:list) {

			        if(method.equalsIgnoreCase("EBR")) {
			        	System.out.println("****EBR Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
				        EBR ebr =new EBR();
				        results = eval.crossValidate(ebr, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			    	    
			        }else if(method.equalsIgnoreCase("ECC")) {
			        	System.out.println("****ECC Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
			        	ECC ecc =new ECC(10);
				        results = eval.crossValidate(ecc, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			    	    
			        }else if(method.equalsIgnoreCase("EPS")) {
			        	System.out.println("****EPS Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
			        	EPS eps =new EPS(10);
				        results = eval.crossValidate(eps, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			    	    
			        }else if(method.equalsIgnoreCase("RAkEL")) {
			        	System.out.println("****RAkEL Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
			        	RAkEL rakel =new RAkEL();
				        results = eval.crossValidate(rakel, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			    	    
			        }else if(method.equalsIgnoreCase("CDE")) {
			        	System.out.println("****CDE Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
			        	EnsembleOfSubsetLearners learner= null;
			        	learner = new EnsembleOfSubsetLearners(new LabelPowerset(new J48()), new J48(), new ConditionalDependenceIdentifier(new J48()), 10);  
						learner.setSeed(10);
			        	learner.setNumOfRandomPartitions(10);		
			        		
				        results = eval.crossValidate(learner, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			    	    
			        }else if(method.equalsIgnoreCase("AdaBoostMH")) {
			        	System.out.println("****AdaBoostMH Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
			        	AdaBoostMH adaboostmh=new AdaBoostMH();	
			        		
				        results = eval.crossValidate(adaboostmh, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			    	    
			        }else if(method.equalsIgnoreCase("MLS")) {
			        	System.out.println("****MLS Result*****");
			        	long time_in = System.currentTimeMillis();
			        	
			        	MultiLabelStacking mls= new MultiLabelStacking();
			        		
				        results = eval.crossValidate(mls, dataset, numFolds);
				        
				        long time_fin = System.currentTimeMillis();
				        System.out.println(results);
				        long total_time = time_fin - time_in;
			    	    System.out.println("Execution time (ms): " + total_time);
			        }
		        } 
	
	 }
		
	}
}
