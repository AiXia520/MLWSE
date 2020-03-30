from skmultilearn.dataset import load_from_arff
import pandas as pd
# dataset:datasets name
# label_count:Number of labels in dataset
# input：The path of the datasets;label_count
# output：X，Y
def read_arff(path,label_count):

    path_to_arff_file=path+".arff"
    arff_file_is_sparse = False
    X, Y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location="end",
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    n_samples, n_features = X.shape
    n_samples, label=Y.shape
    print("n_samples："+str(n_samples)+"  n_features："+str(n_features)+"  label_count："+str(label))
    return(X, Y)

if __name__ == '__main__':
    dataset = "emotions"
    path = "data/" + dataset+ "/" + dataset
    label_count = 53
    X, Y= read_arff(path, label_count)
    print(type(X))
    print(type(Y))
    # print(pd.DataFrame(Y.todense()))
