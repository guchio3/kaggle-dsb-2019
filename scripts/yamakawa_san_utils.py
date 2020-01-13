import logging
import multiprocessing
import os
import pickle
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy as sp
from joblib import Parallel, delayed
from numba import jit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     train_test_split)

import lightgbm as lgb
from lightgbm.callback import _format_eval_result

# import modules
# sys.path.append('../utils/')
# from large_file_pickle import pickle_dump, pickle_load
# # from notifications import send_line_notification, notify_slack
# from memory_optimize import memory_reducer
# from util import *

# from encoders import frequency_encoding


base_path = './mnt/inputs/origin'


# ---------------------------
# util functions
# ---------------------------
def groupings(df, cols, agg_dict, pref='') -> object:
    """
    Returns:
        object:
    """
    group_df = df.groupby(cols).agg(agg_dict)
    group_df.columns = [pref + c[0] + "_" + c[1]
                        for c in list(group_df.columns)]
    group_df.reset_index(inplace=True)

    return group_df


@contextmanager
def timer(name, logger=None):
    """時間計測
    """
    t0 = time.time()
    if logger:
        logger.log(logging.DEBUG, f'[{name}] start')
    else:
        print(f'[{name}] start')
    yield
    if logger:
        logger.log(logging.DEBUG, f'[{name}] done in {time.time() - t0:.0f} s')
    else:
        print(f'[{name}] done in {time.time() - t0:.0f} s')


def get_val_score(y_true, y_pred, obj="RMSE"):
    # RMSE
    if obj == "RMSE":
        val_score = np.sqrt(mean_squared_error(y_true, y_pred))
    elif obj == "QWK":
        val_score = qwk(y_true, y_pred, max_rat=3)
    else:
        raise ValueError("valuation is not defined!")
    return val_score


def memory_reducer(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        print(col)
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def applyParallel(dfGrouped, func):
    """関数の並列処理
    """
    retLst = Parallel(
        n_jobs=multiprocessing.cpu_count(),
        verbose=5)(
        delayed(func)(group) for name,
        group in dfGrouped)
    return pd.concat(retLst)


# ---------------------------
# logging
# ---------------------------


def log_evaluation(logger, period=100, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (
                env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv)
                                for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))

    _callback.order = 10
    return _callback

# ロガーの作成
# logging.basicConfig(level=logging.DEBUG)


def log_output(subject):
    logger = logging.getLogger('main')
    for h in logger.handlers:
        logger.removeHandler(h)

    logger.setLevel(logging.DEBUG)
    now = int(time.time())

    log_dir = os.path.join(os.path.dirname("__file__"), "./mnt/logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = Path(log_dir) / "{}_{}.log".format(subject, now)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)

    return logger, log_path


# ---------------------------
# pickle files
# ---------------------------

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)

        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            # print(n, idx)
            batch_size = min(n - idx, 1 << 31 - 1)
            # print(batch_size)
            # print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            # print("done.", flush=True)
            idx += batch_size
        print("calculate done!")


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


# ---------------------------
# model
# ---------------------------
class LightGBM():
    def __init__(self, param):

        self.predict_type = param["predict_type"]  # classifier, regressor
        self.train_params = param["train_params"]
        self.train_cols = param["train_cols"]
        self.cat_cols = param["cat_cols"]
        self.target = param["target"]
        self.is_debug = param["is_debug"]

    def train(self, train, valid, logger):
        if type(train) != pd.DataFrame or type(valid) != pd.DataFrame:
            raise ValueError(
                'Parameter train and valid must be pandas.DataFrame')

        if list(train.columns) != list(valid.columns):
            raise ValueError('Train and valid must have a same column list')

        trn_x, trn_y = train[self.train_cols], train[self.target]
        val_x, val_y = valid[self.train_cols], valid[self.target]
        callbacks = [log_evaluation(logger, period=500)]

        if self.predict_type == "binary_classifier":
            clf = lgb.LGBMClassifier(**self.train_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                verbose=500,
                early_stopping_rounds=500,
                callbacks=callbacks,
                categorical_feature=self.cat_cols,
            )
            oof = clf.predict_proba(
                val_x, num_iteration=clf.best_iteration_)[:, 1]

        elif self.predict_type == "multi_classifier":
            clf = lgb.LGBMClassifier(**self.train_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                verbose=500,
                early_stopping_rounds=500,
                callbacks=callbacks,
                categorical_feature=self.cat_cols,
                eval_metric=eval_qwk_lgb
            )
            oof = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)

        elif self.predict_type == "regressor":
            clf = lgb.LGBMRegressor(**self.train_params)
            clf.fit(
                trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                verbose=500,
                early_stopping_rounds=500,
                callbacks=callbacks,
                categorical_feature=self.cat_cols,
            )

            oof = clf.predict(val_x, num_iteration=clf.best_iteration_)

        else:
            raise ValueError("unknown prediction type !!")

        self.clf = clf

        # feature importance
        feature_importance_df = pd.DataFrame()
        feature_importance_df["feature"] = self.train_cols
        feature_importance_df["importance"] = self.clf.feature_importances_

        return clf, oof, feature_importance_df

    def predict(self, test, logger):
        if self.predict_type == "classifier":
            prediction = self.clf.predict_proba(test[self.train_cols],
                                                num_iteration=self.clf.best_iteration_)[:, 1]
        elif self.predict_type == "multi_classifier":
            prediction = self.clf.predict_proba(test[self.train_cols],
                                                num_iteration=self.clf.best_iteration_)
        elif self.predict_type == "regressor":
            prediction = self.clf.predict(test[self.train_cols],
                                          num_iteration=self.clf.best_iteration_)
        else:
            raise ValueError("unknown prediction type !!")

        return prediction

    def save_model(self, save_dir):
        pass


# ---------------------------
# metrics
# ---------------------------
@jit
def qwk(y_true, y_pred, max_rat=3):
    y_true_ = np.asarray(y_true, dtype=int)
    y_pred_ = np.asarray(y_pred, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    numerator = 0
    for k in range(y_true_.shape[0]):
        i, j = y_true_[k], y_pred_[k]
        hist1[i] += 1
        hist2[j] += 1
        numerator += (i - j) * (i - j)

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator


def eval_qwk_lgb(y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list],) -> Tuple[str, float, bool]:
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return "qwk", qwk(y_true, y_pred), True


# ---------------------------
# logging
# ---------------------------
class Splitter():
    def __init__(self):
        # self.save_dir = os.path.join(
        #    os.path.dirname("__file__"),
        #    "../data/valid_idx")
        self.idx_list = []
        # os.makedirs(self.save_dir, exist_ok=True)

    def get_kfold_idx(self, split_x, split_y, seed, n_cv=5,
                      stratified=True, group=False, pref=""):
        if group is False:
            if stratified:
                self.folds = StratifiedKFold(
                    n_splits=n_cv, shuffle=False, random_state=seed)
            else:
                self.folds = KFold(
                    n_splits=n_cv, shuffle=True, random_state=seed)

            for i, (trn_, val_) in enumerate(
                    self.folds.split(split_x, split_y)):
                self.idx_list.append([trn_, val_])
                # idx_trn_path = os.path.join(
                #    self.save_dir, "trn_idx_{}{}_{}.npy".format(
                #        pref, i, seed))
                # idx_val_path = os.path.join(
                #    self.save_dir, "val_idx_{}{}_{}.npy".format(
                #        pref, i, seed))
                # np.save(idx_trn_path, trn_)
                # np.save(idx_val_path, val_)

            return self.idx_list

        else:
            groups = split_x
            self.folds = GroupKFold(n_splits=n_cv)
            for i, (trn_, val_) in enumerate(
                    self.folds.split(split_x, split_y, groups)):
                self.idx_list.append([trn_, val_])
                # idx_trn_path = os.path.join(
                #    self.save_dir, "trn_idx_{}{}_{}.npy".format(
                #        pref, i, seed))
                # idx_val_path = os.path.join(
                #    self.save_dir, "val_idx_{}{}_{}.npy".format(
                #        pref, i, seed))
                # np.save(idx_trn_path, trn_)
                # np.save(idx_val_path, val_)

            return self.idx_list


class Validation():
    def __init__(self, validation_param, exp_conf, train, test, logger):
        self.model_name = validation_param["model_name"]
        self.train_small_dataset = exp_conf["train_small_dataset"]
        self.logger = logger
        self.exp_conf = exp_conf
        self.train = self.fix_train_size(train)
        self.test = test
        self.feature_importance = []

        self.logging_valid_parameters()

    def logging_valid_parameters(self):
        self.logger.log(logging.DEBUG, "[use_feature] " + "-" * 50)
        self.logger.log(logging.DEBUG, self.exp_conf["use_feature"])
        self.logger.log(logging.DEBUG, "[train_params] " + "-" * 50)
        self.logger.log(logging.DEBUG, self.exp_conf["train_params"])

    def fix_train_size(self, train):
        if self.train_small_dataset:
            self.logger.log(logging.DEBUG, "Down-sampling train data.")
            self.logger.log(logging.DEBUG, f"Org-shape:{train.shape}")
            p = 0.15  # 学習に使用する割合
            np.random.seed(773)
            int_p = int(len(train.index.values) * p)
            sample_index = np.random.choice(
                train.index.values, int_p, replace=False)  # 重複なし

            train = train.loc[train.index.isin(
                sample_index)].reset_index(drop=True)
            self.logger.log(logging.DEBUG,
                            f"sampled train-shape:{train.shape}")
            return train

        return train

    def generate_model(self, model_conf):
        if self.model_name == "LGBM":
            model = LightGBM(model_conf)
        else:
            raise ValueError("permitted models are [LGBM, ..., ]")
        return model

    def do_valid_kfold(self, model_conf, n_splits=5):
        sp = Splitter()
        target = model_conf["target"]
        split_x = self.train["installation_id"]
        split_y = self.train[target]
        seed = 773
        sp.get_kfold_idx(
            split_x,
            split_y,
            seed,
            n_cv=n_splits,
            stratified=False,
            group=True,
            pref=self.exp_conf["exp_name"])

        oof: ndarray = np.zeros((self.train.shape[0]))
        prediction = np.zeros((self.test.shape[0]))

        clf_list = []

        self.logger.log(logging.DEBUG, "[train cols] " + "-" * 50)
        self.logger.log(logging.DEBUG, model_conf["train_cols"])
        self.validation_scores = []

        for i, (trn_idx, val_idx) in enumerate(sp.idx_list):
            self.logger.log(logging.DEBUG, "-" * 60)
            self.logger.log(logging.DEBUG, f"start training: {i}")

            with timer(f"fold {i}", self.logger):
                train_df, valid_df = self.train.loc[trn_idx], self.train.loc[val_idx]
                model = self.generate_model(model_conf)
                clf, fold_oof, feature_importance_df = model.train(
                    train_df, valid_df, self.logger)
#                 fold_oof_class = fold_oof.argmax(axis = 1)

                fold_prediction = model.predict(self.test, self.logger)
#                 fold_val_score = get_val_score(valid_df[target], fold_oof_class, "QWK")

                # calc validation score using best iteration
#                 self.validation_scores.append(fold_val_score)
#                 self.logger.log(logging.DEBUG, f"fold_val_score: {fold_val_score:,.5f}")

                clf_list.append(clf)
                oof[val_idx] = fold_oof

                prediction += fold_prediction / n_splits

                feature_importance_df["fold"] = i
                self.feature_importance.append(feature_importance_df)

#         self.logger.log(logging.DEBUG,
# f"Total Validation Score: {sum(self.validation_scores) /
# len(self.validation_scores):,.5f}")

        self.feature_importance = pd.concat(self.feature_importance, axis=0)

        return clf_list, oof, prediction, self.feature_importance

    def do_adversarial_valid_kfold(self, model_conf, n_splits=2):
        sp = Splitter()
        target = "is_test"
        split_x = self.train["installation_id"]
        split_y = self.train[target]
        seed = 773
        sp.get_kfold_idx(
            split_x,
            split_y,
            seed,
            n_cv=n_splits,
            stratified=True,
            pref="adv")

        target_length = 1
        oof: ndarray = np.zeros(self.train.shape[0])
        prediction = np.zeros(self.test.shape[0])

        clf_list = []

        self.logger.log(logging.DEBUG, "[train cols] " + "-" * 50)
        self.logger.log(logging.DEBUG, model_conf["train_cols"])
        self.validation_scores = []

        for i, (trn_idx, val_idx) in enumerate(sp.idx_list):
            self.logger.log(logging.DEBUG, "-" * 60)
            self.logger.log(logging.DEBUG, f"start training: {i}")

            with timer(f"fold {i}", self.logger):
                train_df, valid_df = self.train.loc[trn_idx], self.train.loc[val_idx]
                model = self.generate_model(model_conf)
                clf, fold_oof, feature_importance_df = model.train(
                    train_df, valid_df, self.logger)

                # calc validation score using clf.best_iteration_
                fold_val_score = get_val_score(valid_df[target], fold_oof)
                self.validation_scores.append(fold_val_score)
                self.logger.log(logging.DEBUG,
                                f"fold_val_score: {fold_val_score:,.5f}")

                clf_list.append(clf)
                oof[val_idx] = fold_oof

                feature_importance_df["fold"] = i
                self.feature_importance.append(feature_importance_df)

        self.logger.log(logging.DEBUG,
                        f"Total Validation Score: {sum(self.validation_scores) / len(self.validation_scores):,.5f}")

        oof = np.expm1(oof)
        self.train["pred_y"] = oof
        self.feature_importance = pd.concat(self.feature_importance, axis=0)

        return clf_list, oof, prediction, self.feature_importance


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) +
                     [np.inf], labels=[0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y, initial_coef=[0.5, 1.5, 2.5]):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) +
                      [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
