import os
import sys
import random
from pathlib import Path
from datetime import datetime
from dateutil import tz

import numpy as np
import torch

import config
from utils import Logger, seed_worker, log_results

import main as muse_main
import data_parser as dp
from dataset import MuSeDataset, custom_collate_fn
from model import Model
from train import train_model
from eval import evaluate, calc_ccc, flatten_stress_for_ccc


def process_onefold(data, cv_fold, args):

    data_loader = {}
    for partition in data.keys():  # one DataLoader for each partition
        set = MuSeDataset(data, partition)
        batch_size = args.batch_size if partition == 'train' else (1 if args.task=='stress' else 2*args.batch_size)
        shuffle = True if partition == 'train' else False  # shuffle only for train partition
        data_loader[partition] = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                                             worker_init_fn=seed_worker, collate_fn=custom_collate_fn)

    args.d_in = data_loader['train'].dataset.get_feature_dim()

    args.n_targets = config.NUM_TARGETS[args.task]
    args.n_to_1 = args.task in config.N_TO_1_TASKS

    loss_fn, loss_str = muse_main.get_loss_fn(args.task)
    eval_fn, eval_str = muse_main.get_eval_fn(args.task)

    if args.eval_model is None:  # Train and validate for each seed
        seeds = range(args.seed, args.seed + args.n_seeds)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)

            if args.tl_model is not None:
                model_path = Path(args.tl_model)
                print(f'Loading model for transfer learning from {model_path}')
                model = torch.load(model_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

                if args.aggregation_method is None:
                    model.set_n_to_1(True)  # set model for use with MBP single value ground truth
                else:
                    model.out.agg_method = args.aggregation_method

                # freeze layers so only ouput layers are finetuned
                if args.freeze_rnn:
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.out.parameters():
                        param.requires_grad = True

            else:
                model = Model(args)

            print('=' * 50)
            print(f'Training model for CV {cv_fold}... [seed {seed}] for at most {args.epochs} epochs')

            model_path = os.path.join(args.paths['model'], f'cv{cv_fold}')
            os.makedirs(model_path, exist_ok=True)
            val_loss, val_score, best_model_file = train_model(args.task, model, data_loader, args.epochs,
                                                               args.lr, model_path, seed, use_gpu=args.use_gpu,
                                                               loss_fn=loss_fn, eval_fn=eval_fn,
                                                               eval_metric_str=eval_str,
                                                               regularization=args.regularization,
                                                               early_stopping_patience=args.early_stopping_patience,
                                                               reduce_lr_patience=args.reduce_lr_patience)
            # restore best model encountered during training
            model = torch.load(best_model_file)
            if not args.predict:  # run evaluation only if test labels are available
                test_loss, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn,
                                                 eval_fn=eval_fn, use_gpu=args.use_gpu)
                test_scores.append(test_score)
                print(f'[Test {eval_str}]:  {test_score:7.4f}')
            val_losses.append(val_loss)
            val_scores.append(val_score)
            best_model_files.append(best_model_file)

        best_idx = val_scores.index(max(val_scores))  # find best performing seed

        print('=' * 50)
        print(f'Best {eval_str} on [Val] for seed {seeds[best_idx]}: '
              f'[Val {eval_str}]: {val_scores[best_idx]:7.4f}'
              f"{f' | [Test {eval_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}")
        print('=' * 50)

        model_file = best_model_files[best_idx]  # best model of all of the seeds
        if not args.result_csv is None:
            log_results(args.result_csv, params=args, seeds = list(seeds), metric_name=eval_str,
                        model_files=best_model_files, test_results=test_scores, val_results=val_scores,
                        best_idx=best_idx)

    else:  # Evaluate existing model (No training)
        model_file = args.eval_model
        model = torch.load(model_file, map_location=torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
        _, valid_score = evaluate(args.task, model, data_loader['devel'], loss_fn=loss_fn, eval_fn=eval_fn,
                                  use_gpu=args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {eval_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                                     use_gpu=args.use_gpu)
            print(f'[Test {eval_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting test samples...')
        best_model = torch.load(model_file)
        evaluate(args.task, best_model, data_loader['test'], loss_fn=loss_fn, eval_fn=eval_fn,
                 use_gpu=args.use_gpu, predict=True, prediction_path=args.paths['predict'], filename='predictions.csv')

    print('Done.')

def run_cv(args):

    # ensure reproducibility
    np.random.seed(10)
    random.seed(10)

    # emo_dim only relevant for stress
    args.emo_dim = args.emo_dim if args.task in ['stress', 'tl_stress', 'sex_test'] else ''

    # get cv based data from partitions
    print('Loading data ...')
    cv_data = dp.load_data_cv(args.task, args.n_folds, args.paths, args.feature, args.emo_dim, args.normalize,
                              args.normalize_labels, args.win_len, args.hop_len, save=args.cache)
    for cv_fold, data in cv_data:
        args.cv_fold = cv_fold
        process_onefold(data, cv_fold, args)

def main():
    args = muse_main.parse_args(cv=True)

    args.log_file_name = '{}_[{}]_[{}]_[{}_{}_{}_{}]_[{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature, args.emo_dim,
        args.d_rnn, args.rnn_n_layers, args.rnn_bi, args.d_fc_out, args.lr, args.batch_size) if args.task == 'stress' else \
        '{}_[{}]_[{}_{}_{}_{}]_[{}_{}]'.format(datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature.replace(os.path.sep, "-"),
                                                 args.d_rnn, args.rnn_n_layers, args.rnn_bi, args.d_fc_out, args.lr,args.batch_size)

    # adjust your paths in config.py
    args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task),
                  'data': os.path.join(config.DATA_FOLDER, args.task),
                  'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name)}
    if args.predict:
        args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name)
    for folder in args.paths.values():
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    args.paths.update({'features': config.PATH_TO_FEATURES[args.task],
                       'labels': config.PATH_TO_LABELS[args.task],
                       'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(os.path.join(args.paths['log'], args.log_file_name + '.txt'))
    print(' '.join(sys.argv))

    # start up the training
    run_cv(args)

if __name__ == '__main__':
    main()