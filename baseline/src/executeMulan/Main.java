package executeMulan;
import executeMulan.mlc.*;
import executeMulan.mtr.*;
import weka.core.Utils;

public class Main {

	public static void showUse()
	{
		System.out.println("Parameters:");
		
		//Files
		System.out.println("\t -t Train .arff file");
		System.out.println("\t -T Test .arff file");
		System.out.println("\t -x  Labels .xml file");
		
		//Algorithm
		System.out.println("\t -a Algorithm:");
			//Classification algorithms
		System.out.println("\t\tClassification algorithms:");
		System.out.println("\t\t\tAdaBoostMH -> AdaBoostMH");
		System.out.println("\t\t\tBPMLL -> Back-Propagation for MLL");
		System.out.println("\t\t\tBR -> Binary Relevance");
		System.out.println("\t\t\tCC -> Classifier Chains");
		System.out.println("\t\t\tCDE -> Chi-Dep Ensemble");
		System.out.println("\t\t\tCLR -> Calibrated Label Ranking");
		System.out.println("\t\t\tEBR -> Ensemble of Binary Relevance");
		System.out.println("\t\t\tECC -> Ensemble of Classifier Chains");
		System.out.println("\t\t\tEPS -> Ensemble of Pruned Sets");
		System.out.println("\t\t\tHOMER -> HOMER");
		System.out.println("\t\t\tIBLR -> Instance-Based Logistic Regression");
		System.out.println("\t\t\tLP -> Label Powerset");
		System.out.println("\t\t\tLPBR -> LPBR");
		System.out.println("\t\t\tMLkNN -> ML k-Nearest Neighbours");
		System.out.println("\t\t\tMLS -> Multi-Label Stacking");
		System.out.println("\t\t\tPS -> Pruned Sets");
		System.out.println("\t\t\tRAkEL -> RAkEL");	
		System.out.println("\t\t\tRFPCT -> Random Forest PCTs");	
			//Regression algorithms
		System.out.println("\t\tRegression algorithms:");
		System.out.println("\t\t\tERC -> Ensemble of Regressor Chains");
//		System.out.println("\t\t\tMORF -> Multi-Output Random Forest");
		System.out.println("\t\t\tRC -> Regressor Chains");
		System.out.println("\t\t\tRLC -> Random Linear Combinations Normalized");
		System.out.println("\t\t\tST -> Single Target");
		System.out.println("\t\t\tSST -> Stacked ST");
		
		//Number of seed numbers == Number of executions
		System.out.println("\t -i Number of random seeds");
		
		//Show or not the measures (macro) for each label
		System.out.println("\t -l 1/0 Show (macro) measures for each label (1) or not (0)");
		
		//Output file
		System.out.println("\t -o Output file");
	}
	

	public static void main(String [] args)
	{		
		String tvalue=null, Tvalue=null, xvalue=null, avalue=null, ovalue=null, lvalueStr=null, ivalueStr=null;
		
		//By default, macro measures are not shown for each label
		boolean lvalue = false;
		//By default, 10 random seeds are used
		int ivalue = 10;
		
		try {
//			tvalue = Utils.getOption("t", args);
//			Tvalue = Utils.getOption("T", args);
//			xvalue = Utils.getOption("x", args);
//			avalue = Utils.getOption("a", args);
//			ovalue = Utils.getOption("o", args);
			
			
			tvalue="src/data/emotions/emotions-train1.arff";
			Tvalue="src/data/emotions/emotions-test1.arff";
			xvalue="src/data/emotions/emotions.xml";
			ovalue="AdaBoostMH";
			ExecuteAdaBoostMH a = new ExecuteAdaBoostMH();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
			
			if((tvalue.length() == 0) || (Tvalue.length() == 0) || (xvalue.length() == 0) || (avalue.length() == 0) || (ovalue.length() == 0))
			{
				if(tvalue.length() == 0) {
					System.out.println("Please enter the train dataset filename.");
				}
				if(Tvalue.length() == 0) {
					System.out.println("Please enter the test dataset filename.");
				}
				if(xvalue.length() == 0) {
					System.out.println("Please enter the xml dataset filename.");
				}
				if(avalue.length() == 0) {
					System.out.println("Please enter the algorithm.");
				}
				if(ovalue.length() == 0) {
					System.out.println("Please enter the output filename.");
				}			
				
				showUse();
				System.exit(1);
			}
			
			lvalueStr = Utils.getOption("l", args);
			if(lvalueStr.length() != 0) {
				if(lvalueStr.equalsIgnoreCase("1")) {
					lvalue = true;
				}
				else {
					lvalue = false;
				}
			}
			
			ivalueStr = Utils.getOption("i", args);
			if(ivalueStr.length() != 0) {
				ivalue = Integer.parseInt(ivalueStr);
			}
		} catch (Exception e) {
			showUse();
			System.exit(1);
//			e.printStackTrace();
		}
		
		if(avalue.equalsIgnoreCase("AdaBoostMH"))
		{
			ExecuteAdaBoostMH a = new ExecuteAdaBoostMH();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("BPMLL"))
		{
			ExecuteBPMLL a = new ExecuteBPMLL();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("BR"))
		{			
			ExecuteBR a = new ExecuteBR();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("CC"))
		{
			ExecuteCC a = new ExecuteCC();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("CLR"))
		{
			ExecuteCLR a = new ExecuteCLR();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("EBR"))
		{
			ExecuteEBR a = new ExecuteEBR();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("ECC"))
		{
			ExecuteECC a = new ExecuteECC();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("EPS"))
		{
			ExecuteEPS a = new ExecuteEPS();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("CDE"))
		{
			ExecuteCDE a = new ExecuteCDE();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("HOMER"))
		{
			ExecuteHOMER a = new ExecuteHOMER();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("IBLR"))
		{
			ExecuteIBLR a = new ExecuteIBLR();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("LP"))
		{
			ExecuteLP a = new ExecuteLP();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("LPBR"))
		{
			ExecuteLPBR a = new ExecuteLPBR();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("MLKNN"))
		{
			ExecuteMLkNN a = new ExecuteMLkNN();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("MLS"))
		{
			ExecuteMLS a = new ExecuteMLS();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("PS"))
		{
			ExecutePS a = new ExecutePS();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.toUpperCase().equalsIgnoreCase("RAKEL"))
		{
			ExecuteRAkEL a = new ExecuteRAkEL();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("RFPCT"))
		{
			ExecuteRFPCT a = new ExecuteRFPCT();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("ERC"))
		{
			ExecuteERC a = new ExecuteERC();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("MORF"))
		{
			ExecuteMORF a = new ExecuteMORF();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("RC"))
		{
			ExecuteRC a = new ExecuteRC();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("RLC"))
		{
			ExecuteRLC a = new ExecuteRLC();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue, ivalue);
		}
		else if(avalue.equalsIgnoreCase("ST"))
		{
			ExecuteST a = new ExecuteST();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else if(avalue.equalsIgnoreCase("SST"))
		{
			ExecuteSST a = new ExecuteSST();
			a.execute(tvalue, Tvalue, xvalue, ovalue, lvalue);
		}
		else
		{
			showUse();
			System.out.println("The algorithm \'" + avalue + "\' is not defined");
			System.exit(-1);
		}
	}
	
}
