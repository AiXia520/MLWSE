package executeMulan;

import java.io.BufferedReader;

import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.clusterers.SimpleKMeans;
public class Mlknn 
{
	public static void main(String [] args) throws Exception
	{
		String input ="";
		MultiLabelInstances dataset = null;
		MultiLabelInstances dataset2 = null;
		 
		String arff_train = input+"bibtex0.arff";
		String xml = input+"bibtex.xml";
		
		try 
		{
			dataset = new MultiLabelInstances(arff_train, xml);		
		}
		catch(InvalidDataFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		MLkNN learner1 = new MLkNN(8,1.0); 
		
		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
				
		int numFolds=10;
		
		results = eval.crossValidate(learner1, dataset, numFolds);
		System.out.println(results);	

		
	}
}



