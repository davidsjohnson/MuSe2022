import os
import types

import data_parser as dp
import config

class TestDataLoading:

    def test_data_gen(self):
        task = 'tl_stress'
        feature = 'egemaps'
        emo_dim = 'Delta_LNsAA'
        normalize = True
        win_len = 200
        hop_len = 100
        n_folds = 5
        paths = {'log': os.path.join(config.LOG_FOLDER, task),
                 'data': os.path.join(config.DATA_FOLDER, task),
                 'model': os.path.join(config.MODEL_FOLDER, task, 'log_file'),
                 'features': config.PATH_TO_FEATURES[task],
                 'labels': config.PATH_TO_LABELS[task],
                 'partition': config.PARTITION_FILES[task],}
        cv_data = dp.load_data_cv(task, n_folds, paths, feature, emo_dim, normalize,
                                  win_len, hop_len, save=True)
        assert isinstance(cv_data, types.GeneratorType)

        cv_data = list(cv_data)

        # check that data is as expected
        assert len(cv_data) == n_folds
        assert cv_data[n_folds-1][0] == n_folds
        assert isinstance(cv_data[n_folds-1][1], dict)
        # TODO some better tests here


class TestCrossVal:

    def test_get_partition_cvs(self):
        task = 'tl_stress'
        n_folds = 5
        n_subs = 58
        parts = ['train', 'test', 'devel']
        partition_file = config.PARTITION_FILES[task]

        for cv_fold, (sub2part, part2sub) in enumerate(dp.get_data_partition_allfolds(partition_file, n_folds=n_folds), start=1):
            assert len(sub2part.keys()) == n_subs
            assert list(part2sub.keys()).sort() == parts.sort()
        assert cv_fold == n_folds

    def test_no_cv_testoverlap(self):
        task = 'tl_stress'
        n_folds = 5
        partition_file = config.PARTITION_FILES[task]

        prev_testsubs = []
        for cv_fold, (sub2part, part2sub) in enumerate(dp.get_data_partition_allfolds(partition_file, n_folds=n_folds), start=1):
            testsubs = part2sub['test']
            overlap =  set(prev_testsubs) & set(testsubs)
            prev_testsubs.extend(testsubs)

            assert not bool(overlap), f'Overlap in Fold {cv_fold}, with subs {overlap}'