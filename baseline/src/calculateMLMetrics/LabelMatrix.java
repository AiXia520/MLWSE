package calculateMLMetrics;

public class LabelMatrix {
	
	public int [][] realLabels;
	public int [][] predLabels;
	
	public LabelMatrix(int nInstances, int nLabels) {
		realLabels = new int[nInstances][nLabels];
		predLabels = new int[nInstances][nLabels];
	}

}
