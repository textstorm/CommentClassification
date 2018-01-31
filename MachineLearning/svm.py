import time
import numpy as np
import pandas as pd

from sklearn import svm
from data import DataLoader
from sklearn.model_selection import GridSearchCV

def main():
    dl = DataLoader()
    model = svm.SVC(probability=True)

    result = np.zeros((dl.data['submit']['X'].shape[0], dl.data['train']['Y'].shape[1]))
    for i in range(len(dl.params['class_list'])):
        t0 = time.time()
        model.fit(dl.data['train']['X'], dl.data['train']['Y'][:, i])
        print("%.2f secs ==> [%d/6]LogisticRegression().fit()" % (time.time()-t0, i+1))
        result[:, i] = model.predict_proba(dl.data['submit']['X'])[:, 1]

    submit = pd.read_csv("../data/sample_submission.csv")
    submit[dl.params['class_list']] = result
    submit.to_csv('submission-svm.csv', index=False)
    print('End')

def fine_tune():
    dl = DataLoader()
    grid = GridSearchCV(svm.SVC(), param_grid={"C":[0.1, 0.5, 1, 5, 10], "gamma": [1, 0.1, 0.01, 0.001, 0.0001]}, cv=4)
    grid.fit(dl.data['train']['X'], dl.data['train']['Y'][:, 0])
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

if __name__ == '__main__':
    # main()
    fine_tune()
