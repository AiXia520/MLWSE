package mulan.evaluation.measure;

public class MacroAccuracy extends LabelBasedAccuracy implements
		MacroAverageMeasure {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8790128844921277L;

	/**
     * Constructs a new object with given number of labels
     * 
     * @param numOfLabels the number of labels
     */
    public MacroAccuracy(int numOfLabels) {
        super(numOfLabels);
    }

    @Override
    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {

            if (truePositives[labelIndex] + trueNegatives[labelIndex] == 0) {
                sum+=0;
            }
            else
            {
            	sum+=((double)(truePositives[labelIndex] + trueNegatives[labelIndex]) / (truePositives[labelIndex] + falsePositives[labelIndex] + falseNegatives[labelIndex] + trueNegatives[labelIndex]));
            }

            count++;
        }
        return sum / count;
    }

    @Override
    public String getName() {
        return "Macro-averaged Accuracy";
    }

    /**
     * Returns the precision for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the precision for the given label
     */
    @Override
    public double getValue(int labelIndex) {
    	if (truePositives[labelIndex] + trueNegatives[labelIndex] == 0) {
            return 0;
        }
        else
        {
        	return (truePositives[labelIndex] + trueNegatives[labelIndex]) / (truePositives[labelIndex] + falsePositives[labelIndex] + falseNegatives[labelIndex] + trueNegatives[labelIndex]);
        }
    }

}
