package executeMulan;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.*;
import mulan.evaluation.measures.regression.example.ExampleBasedRMaxSE;
import mulan.evaluation.measures.regression.macro.MacroMAE;
import mulan.evaluation.measures.regression.macro.MacroMaxAE;
import mulan.evaluation.measures.regression.macro.MacroRMSE;
import mulan.evaluation.measures.regression.macro.MacroRMaxSE;
import mulan.evaluation.measures.regression.macro.MacroRelMAE;
import mulan.evaluation.measures.regression.macro.MacroRelRMSE;


public class ExecuteMulanAlgorithm {
	
	 public PrintWriter pw = null;
	 public MultiLabelInstances trainingSet = null;
	 public MultiLabelInstances testSet = null;
	 public Evaluator eval = new Evaluator();
	 public Evaluation results;
	 public List<Measure> measures = new ArrayList<Measure>();
	 public long time_in, time_fin, total_time;

	public void prepareExecution(String tvalue, String Tvalue, String xvalue, String ovalue) {
		try {
			trainingSet = new MultiLabelInstances(tvalue, xvalue);
			testSet = new MultiLabelInstances(Tvalue, xvalue);
			
			pw = new PrintWriter(new FileWriter(ovalue, true));
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void printHeader(boolean lvalue) throws Exception{
		//Print header
	    pw.print("Dataset" + ";");
        for(Measure m : measures)
        {
        	pw.print(m.getName() + ";");
        	if((lvalue) && (m.getClass().getName().contains("Macro")))
            {
        		for(int l=0; l<trainingSet.getNumLabels(); l++)
                {
        			pw.print(m.getName() + " - " + trainingSet.getLabelNames()[l] + ";");
                }
            }
        }
        pw.print("Execution time (ms): ");
        pw.println();
	}
	
	public void printResults(String tvalue, boolean lvalue, String algorithm) throws Exception {
		String [] p = tvalue.split("\\/");
		String datasetName = p[p.length-1].split("\\.")[0];                   
        pw.print(algorithm + "_" + datasetName + ";");
        
        for(Measure m : results.getMeasures())
        {
        	pw.print(m.getValue() + ";");
     	   	if((lvalue) && (m.getClass().getName().contains("Macro")))
            {
     	   		for(int l=0; l<trainingSet.getNumLabels(); l++)
     	   		{
     	   			pw.print(((MacroAverageMeasure) m).getValue(l) + ";");
     	   		}
            }
        }
        pw.print(total_time + ";");
        pw.println();    
	}
	
	public void execute(String tvalue, String Tvalue, String xvalue, String ovalue, boolean lvalue) {
		System.out.println("Method not implemented");
		System.exit(-1);
	}
	
	protected static List<Measure> prepareMeasuresClassification(MultiLabelInstances mlTrainData) {
        List<Measure> measures = new ArrayList<Measure>();

        int numOfLabels = mlTrainData.getNumLabels();
        
        // add example-based measures
        measures.add(new HammingLoss());
        measures.add(new SubsetAccuracy());
        measures.add(new ExampleBasedPrecision());
        measures.add(new ExampleBasedRecall());
        measures.add(new ExampleBasedFMeasure());
        measures.add(new ExampleBasedAccuracy());
        measures.add(new ExampleBasedSpecificity());
        
        // add label-based measures
        measures.add(new MicroPrecision(numOfLabels));
        measures.add(new MicroRecall(numOfLabels));
        measures.add(new MicroFMeasure(numOfLabels));
        measures.add(new MicroSpecificity(numOfLabels));
        //measures.add(new MicroAccuracy(numOfLabels));
        measures.add(new MacroPrecision(numOfLabels));
        measures.add(new MacroRecall(numOfLabels));
        measures.add(new MacroFMeasure(numOfLabels));
        measures.add(new MacroSpecificity(numOfLabels));
        //measures.add(new MacroAccuracy(numOfLabels));
        
        // add ranking based measures
        measures.add(new AveragePrecision());
        measures.add(new Coverage());
        measures.add(new OneError());
        measures.add(new IsError());
        measures.add(new ErrorSetSize());
        measures.add(new RankingLoss());
        
        // add confidence measures if applicable
        measures.add(new MeanAveragePrecision(numOfLabels));
        measures.add(new GeometricMeanAveragePrecision(numOfLabels));
//        measures.add(new MicroAUC(numOfLabels));
//        measures.add(new MacroAUC(numOfLabels));

        return measures;
    }
	
	protected static List<Measure> prepareMeasuresRegression(MultiLabelInstances mlTrainData, MultiLabelInstances mlTestData) {
        List<Measure> measures = new ArrayList<Measure>();

        int numOfLabels = mlTrainData.getNumLabels();
        measures.add(new MacroMAE(numOfLabels));
        measures.add(new MacroRMSE(numOfLabels));
        measures.add(new MacroRelMAE(mlTrainData, mlTestData));
        measures.add(new MacroRelRMSE(mlTrainData, mlTestData));
        
        measures.add(new MacroMaxAE(numOfLabels));
        measures.add(new MacroRMaxSE(numOfLabels));
        
        measures.add(new ExampleBasedRMaxSE());

        return measures;
    }

}
