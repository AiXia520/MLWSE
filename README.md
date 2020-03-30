# Multi-Label Classification with Weighted Classifier Selection and Stacked Ensemble

The code for the paper Multi-Label Classification with Weighted Classifier Selection and Stacked Ensemble

## Requirements

* python 3.6
* [scikit-learn](https://scikit-learn.org/)
* [scikit-multilearn](http://scikit.ml/api/skmultilearn.html)
* numpy
* scipy
* pandas

## Instructions

### Materials Preparation

There is a folder `data/`, which contains some multi-label datasets.These datasets can be download from the websites of [Mulan](http://mulan.sourceforge.net/), [KDIS](http://www.uco.es/kdis/mllresources/), and [Meka](http://waikato.github.io/meka/datasets/).

#### Baseline(Ensemble multi-label classification methods)
1. There is a folder `baseline/`, which contains seven state-of-the-art ensemble multi-label classification methods including EBR, ECC, EPS, RAkEL, CDE, AdaBoost.MH and MLS. These methods were implemented using the Mulan and Meka frameworks, which provide an API to use their functionalities in Java code. You can also download from [KDIS-lib](https://github.com/kdis-lab/ExecuteMulan).
2. We have make a detailed integration, You can run `main.java` under the package `src/com.baseline.ensmeble` to get the experimental results.

#### Read multi-label datasets
1. `cd root/`;
2. When you use multi-label datasets in [arff](https://pypi.org/project/arff/0.9/) format, you can run `python read_arff.py`;
3. When you use multi-label datasets in mat format, you can run `python read_matfile.py`;

### MLWSE-L1 And MLWSE-L21
we use the accelerated proximal gradient and block coordinate descent to optimize MLWSE-L1 and MLWSE-L21, respectively.You can find in `lasso.py` and `util/blockwise_descent_semisparse.py`.

#### 2-D Synthetic Datasets Results

With 2-D synthetic datasets, we evaluate the weighted classifier selection ability of our approach by gradually adding different technical components. You can find in `simulation/simulation_lasso.py`
1. `cd root/`;
2. Run `lasso_stacking_simulaiton.py`, get `2-D Synthetic Datasets Results`;

#### Benchmark Datasets Results
For multi-label benchmark datasets,you can use the following steps:
1. `cd root/`;
2. Run `python lasso_stacking.py`, get `Benchmark Datasets MLWSE-L1 results`;
3. Run `python lasso_stacking2.py`, get `Benchmark Datasets MLWSE-L21 results`;

#### Real-world Application Results
To explore the potential application of our proposed method,  we apply our approach to a real Cardiovascular and Cerebrovascular Disease (CCD) dataset to demonstrate its potential for practical applications in medical diagnosis, and we take CCD dataset as another benchmarking dataset to run the experiments. 

1. `cd root/`;
2. Run `python lasso_stacking_ccd.py`, get `CCD datasets MLWSE-L1 results`;
3. Run `python lasso_stacking2_ccd.py`, get `CCD datasets MLWSE-L21 results`;


### Algorithm Evaluation

#### Friedman Statistics Evaluation


#### Parameter Sensitivity Evaluation 


#### Convergence Evaluation













