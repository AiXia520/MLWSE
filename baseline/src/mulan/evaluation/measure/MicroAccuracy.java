package mulan.evaluation.measure;

import weka.core.Utils;

public class MicroAccuracy extends LabelBasedAccuracy {

    /**
	 * 
	 */
	private static final long serialVersionUID = -7064027048632761031L;

	/**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public MicroAccuracy(int numOfLabels) {
        super(numOfLabels);
    }

    @Override
    public double getValue() {
        double tp = Utils.sum(truePositives);
        double fp = Utils.sum(falsePositives);
        double fn = Utils.sum(falseNegatives);
        double tn = Utils.sum(trueNegatives);
        
        if (tp + fp + fn + tn == 0) {
            return 1;
        }
        if (tp + tn == 0) {
            return 0;
        }
        return (tp + tn) / (tp + fp + fn + tn);
    }

    @Override
    public String getName() {
        return "Micro-averaged Accuracy";
    }
}
