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
2. We have done a detailed integration, You can run `main.java` under the package `src/com.baseline.ensmeble` to get the experimental results.

#### Read multi-label datasets
1. `cd root/`
2. When you use multi-label datasets in arff format, you can run `read_arff.py`;
3. When you use multi-label datasets in mat format, you can run `read_matfile.py`;

### MLWSE-L1 And MLWSE-L21
1. `cd root/`
2. Run `python lasso_stacking.py`, get `MLWSE-L1 results`
3. Run `python lasso_stacking2.py`, get `MLWSE-L21 results`









