package mulan.evaluation.measure;

public abstract class LabelBasedAccuracy extends LabelBasedBipartitionMeasureBase {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3159328315306708402L;

	/**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public LabelBasedAccuracy(int numOfLabels) {
        super(numOfLabels);
    }

    @Override
    public double getIdealValue() {
        return 1;
    }

}
