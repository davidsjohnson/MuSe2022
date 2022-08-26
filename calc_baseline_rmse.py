import os

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor

import data_parser as dp
from eval import rmse
import config

class DummyRegressor2:

    def __init__(self):
        self.y_mean = None

    def fit(self, X, y):
        y = np.array(y)
        self.y_mean = y.mean()

    def predict(self, X):
        if self.y_mean is not None:
            return np.array([self.y_mean] * len(X))
        else:
            raise RuntimeError('Model is not trained.  Run fit before calling predict.')


def main():
    task = 'tl_stress'
    n_folds = 5
    feature = 'egemaps'
    emo_dim = 'Delta_LNsAA'
    normalize = True
    normalize_labels = True
    win_len = 200
    hop_len = 100

    paths = {'log': os.path.join(config.LOG_FOLDER, task),
             'data': os.path.join(config.DATA_FOLDER, task),
             'model': os.path.join(config.MODEL_FOLDER, task, 'log_file'),
             'features': config.PATH_TO_FEATURES[task],
             'labels': config.PATH_TO_LABELS[task],
             'partition': config.PARTITION_FILES[task], }

    cv_data = dp.load_data_cv(task, n_folds, paths, feature, emo_dim, normalize,
                              win_len, hop_len, save=False)

    dfs = []
    for cv_fold, data in cv_data:

        X_train = data['train']['feature']
        y_train = data['train']['label']

        X_devel = data['devel']['feature']
        y_devel = data['devel']['label']

        X_test = data['test']['feature']
        y_test = data['test']['label']

        # train model
        model = DummyRegressor(strategy="mean")
        # model = DummyRegressor2()
        model.fit(X_train, y_train)

        # get predictions for all datasets
        y_train_pred = model.predict(X_train)
        y_devel_pred = model.predict(X_devel)
        y_test_pred = model.predict(X_test)

        # calculate metrics
        rmse_train = rmse([y_train_pred], [y_train])
        rmse_devel = rmse([y_devel_pred], [y_devel])
        rmse_test = rmse([y_test_pred], [y_test])

        res_dict = {
            'rmse': [rmse_train, rmse_devel, rmse_test],
            'partition': ['train', 'devel', 'test'],
            'cv_fold': [cv_fold, cv_fold, cv_fold],
        }

        dfs.append(pd.DataFrame(data=res_dict))

    df_res = pd.concat(dfs)
    df_res.to_csv('results/baseline_results2.csv', index=False)


if __name__ == '__main__':
    main()
