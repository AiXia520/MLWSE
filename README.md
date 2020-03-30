# Multi-Label Classification with Weighted Classifier Selection and Stacked Ensemble

The code for the paper Multi-Label Classification with Weighted Classifier Selection and Stacked Ensemble

## Requirements

* python 3
* [scikit-learn](https://scikit-learn.org/)
* [scikit-multilearn](http://scikit.ml/index.html)
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [pandas](https://pandas.pydata.org/)

## Instructions

### Materials Preparation

There is a folder `data/`, which contains some multi-label datasets. You can download these datasets from the websites of [Mulan](http://mulan.sourceforge.net/), [KDIS](http://www.uco.es/kdis/mllresources/), and [Meka](http://waikato.github.io/meka/datasets/).

#### Baseline(Ensemble multi-label classification methods)
1. There is a folder `baseline/`, which contains seven state-of-the-art ensemble multi-label classification methods including EBR, ECC, EPS, RAkEL, CDE, AdaBoost.MH and MLS. These methods were implemented using the Mulan and Meka frameworks, which provide an API to use their functionalities in Java code. You can also download from [KDIS-lib](https://github.com/kdis-lab/ExecuteMulan).
2. We have make a detailed integration, You can run `Main.java` under the package `src/com.baseline.ensmeble` to get the experimental results.

Run `Main.java`, get `baseline results`;

#### Read multi-label datasets
The multi-label datasets are stored primarily in the [arff](https://pypi.org/project/arff/0.9/) and mat formats, and you can use the following steps:
1. `cd root/`;
2. When you use multi-label datasets in [arff](https://pypi.org/project/arff/0.9/) format, you can run `python read_arff.py`;
3. When you use multi-label datasets in mat format, you can run `python read_matfile.py`;

### MLWSE-L1 And MLWSE-L21
we use the accelerated proximal gradient and block coordinate descent to optimize MLWSE-L1 and MLWSE-L21, respectively.You can find in `lasso.py` and `util/blockwise_descent_semisparse.py`.

#### 2-D Synthetic Datasets Results

With 2-D synthetic datasets, we evaluate the weighted classifier selection ability of our approach by gradually adding different technical components. You can find in `simulation/simulation_lasso.py`, and you can use the following steps:
1. `cd root/`;
2. Run `python lasso_stacking_simulaiton.py`, get `2-D Synthetic Datasets Results`;

#### Benchmark Datasets Results
For multi-label benchmark datasets, you can use the following steps:
1. `cd root/`;
2. Run `python lasso_stacking.py`, get `Benchmark Datasets MLWSE-L1 results`;
3. Run `python lasso_stacking2.py`, get `Benchmark Datasets MLWSE-L21 results`;

#### Real-world Application Results
To explore the potential application of our proposed method,  we apply our approach to a real Cardiovascular and Cerebrovascular Disease (CCD) dataset to demonstrate its potential for practical applications in medical diagnosis, and we take CCD dataset as another benchmarking dataset to run the experiments. You can download the CCD dataset from `data/matfile/ccd.mat`.

1. `cd root/`;
2. Run `python lasso_stacking_ccd.py`, get `CCD datasets MLWSE-L1 results`;
3. Run `python lasso_stacking2_ccd.py`, get `CCD datasets MLWSE-L21 results`;

### Algorithm Evaluation
we used six common evaluation metrics to verify the performance including Hamming loss, Accuracy, Ranking loss, F1, Macro-F1 and Micro-F1. These evaluation metrics have been implemented in `mlmetrics.py`. 

#### Friedman Statistics Evaluation
We employed the Friedman test to statistically analyze the performance of the different algorithms systematically. The detailed Friedman statistics analysis can be found in `result/result_analysis.xls`.  You can run `plot_friedman.py` to get CD diagrams of the algorithms under each evaluation criterion. 

1. `cd root/`;
2. Run `python plot_friedman.py`, get ` CD diagrams of the algorithms`;

#### Parameter Sensitivity Evaluation 
We analyzed the parameter sensitivity of MLWSE-L1 and MLWSE-L21 by conducting experiments on the Emotions and GpositiveGO datasets. The detailed results can be found in `result/stacking_tune_parameter.xls` and `result/stacking_tune_parameter2.xls`. You can use the following steps:
1. `cd root/`;
2. Run `python lasso_stacking_tune_parameter.py`, get `parameter sensitivity results`;
3. Run `python lasso_stacking2_tune_parameter.py`, get `parameter sensitivity results`;

#### Convergence Evaluation
We analysis convergence of MLWSE-L1 and MLWSE-L21 by conducting experiments on the Emotions, Scene, Yeast and VirusGO datasets. The detailed results can be found in `result/stacking_iter_loss.xls` and `result/stacking2_iter_loss.xls`. You can also get results by using the following steps:
1. `cd root/`;
2. Run `python lasso_stacking.py`, get `MLWSE-L1 iter loss `;
3. Run `python lasso_stacking2.py`, get `MLWSE-L21 iter losss`;

## References
[1] G. Tsoumakas, E. Spyromitros-Xioufis, J. Vilcek, I. Vlahavas, Mulan: A java library for multi-label learning, Journal of Machine Learning Research, 12 (2011) 2411-2414.

[2] J. Read, P. Reutemann, B. Pfahringer, G. Holmes, Meka: a multi-label/multi-target extension to weka, Journal of Machine Learning Research, 17 (2016) 667-671.

[3] https://github.com/kdis-lab/ExecuteMulan.

[4] http://scikit.ml/index.html






