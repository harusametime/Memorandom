## Generating confusion matrix in Leave One Out Cross Validation

```python

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

target_label =['A','B','C']
classifier = KNeighborsClassifier(n_neighbors)
conf_mat = pd.DataFrame(0, index=target_label, columns=target_label)
        
loo = LeaveOneOut()
for train_i, test_i in loo.split(X):
    X_train = X[train_i]
    y_train = y[train_i]
    (X_train, y_train) = random_under_sampling(X_train, y_train)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X[test_i])
            
    # Because y_predict is a list consisting of one prediction,
    # the 1st element is counted for evaluation.
    conf_mat[y_predict[0]][y[test_i][0]] +=1
```


## Dealing with imbalanced data (Rundom Under Sampling)


```python

import numpy as np
import pandas as pd
def random_under_sampling(X, y):
    data = pd.DataFrame(X)
    data['label']= y
    labels = np.unique(y, return_counts=True)
    min_number = np.amin(labels[1])
    sampled_data = pd.DataFrame()
    for i in range(len(labels[0])):
        if labels[1][i] > min_number:
            indices = data[data.label == labels[0][i]].index
            random_indices = np.random.choice(indices, min_number, replace=False)
            sampled_data = pd.concat([sampled_data, data.loc[random_indices]])
        else:
            sampled_data = pd.concat([sampled_data, data[data.label == labels[0][i]]])
        #random_indices = 
    nparray_sampled = sampled_data.as_matrix()
    return nparray_sampled[:,0:X.shape[1]], nparray_sampled[:,X.shape[1]]
```
