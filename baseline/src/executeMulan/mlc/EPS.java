package executeMulan.mlc;
import java.util.Random;

import mulan.classifier.transformation.EnsembleOfPrunedSets;


public class EPS extends EnsembleOfPrunedSets {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8232684670814787107L;

	private void setRand(Random r)
	{
		this.rand = r;
	}
	
	public void setSeed(long s)
	{
		Random r = new Random(s);
		setRand(r);
	}
	
	EPS()
	{
		super();
	}
	
	public EPS(int s)
	{
		super();
		Random r = new Random(s);
		setRand(r);
	}
}
