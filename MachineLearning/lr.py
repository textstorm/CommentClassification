import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data import DataLoader


def main():
    dl = DataLoader()
    model = LogisticRegression()
    
    result = np.zeros((dl.data['submit']['X'].shape[0], dl.data['train']['Y'].shape[1]))
    for i in range(len(dl.params['class_list'])):
        t0 = time.time()
        model.fit(dl.data['train']['X'], dl.data['train']['Y'][:, i])
        print("%.2f secs ==> [%d/6]LogisticRegression().fit()" % (time.time()-t0, i+1))
        result[:, i] = model.predict_proba(dl.data['submit']['X'])[:, 1]

    submit = pd.read_csv("../data/sample_submission.csv")
    submit[dl.params['class_list']] = result
    submit.to_csv('submission-lr.csv', index=False)
    print('End')

def fine_tune():
    dl = DataLoader()
    X_train, X_test, y_train, y_test = train_test_split(
        dl.data['train']['X'], dl.data['train']['Y'][0], test_size=0.2, random_state=2018)
    print(dl.data['train']['Y'].shape)
    # for c in [0.1, 1, 10, 100]:
    model = LogisticRegression(C=c)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print(labels[:10])

if __name__ == '__main__':
    # main()
    fine_tune()