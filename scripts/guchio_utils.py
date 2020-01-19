import logging

import numpy as np
import pandas as pd

from yamakawa_san_utils import (LightGBM, OptimizedRounder, Splitter,
                                get_val_score, qwk, timer)

base_path = './mnt/inputs/origin'


# ---------------------------
# util functions
# ---------------------------
class guchioValidation():
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

        optimizers = []
        valid_qwks = []

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

                optR = OptimizedRounder()
                optR.fit(fold_oof, valid_df[target])
                coefficients = optR.coefficients()
                opt_preds = optR.predict(fold_oof, coefficients)
                fold_qwk = qwk(valid_df[target], opt_preds)
                optimizers.append(optR)
                valid_qwks.append(fold_qwk)

                clf_list.append(clf)
                oof[val_idx] = fold_oof

                prediction += fold_prediction / n_splits

                feature_importance_df["fold"] = i
                self.feature_importance.append(feature_importance_df)

#         self.logger.log(logging.DEBUG,
# f"Total Validation Score: {sum(self.validation_scores) /
# len(self.validation_scores):,.5f}")

        self.feature_importance = pd.concat(self.feature_importance, axis=0)

        return clf_list, oof, prediction, self.feature_importance, optimizers, valid_qwks

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

        optimizers = []
        valid_qwks = []

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

                optR = OptimizedRounder()
                optR.fit(fold_oof, valid_df[target])
                coefficients = optR.coefficients()
                opt_preds = optR.predict(fold_oof, coefficients)
                fold_qwk = qwk(valid_df[target], opt_preds)
                optimizers.append(optR)
                valid_qwks.append(fold_qwk)

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

        return clf_list, oof, prediction, self.feature_importance, optimizers, valid_qwks


class Validation2():
    def __init__(self, validation_param, exp_conf, train,
                 test, logger=None, another_train=None):
        self.model_name = validation_param["model_name"]
        self.train_small_dataset = exp_conf["train_small_dataset"]
        self.logger = logger
        self.exp_conf = exp_conf
        self.train = self.fix_train_size(train)
        self.test = test
        self.another_train = another_train
        if another_train:
            self.another_train_idx = np.arange(
                len(another_train)) + len(self.train)
            self.another_train.index = self.another_train_idx
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

    def do_valid_kfold(self, model_conf, n_splits=5,
                       trn_mode='simple', val_mode='simple'):
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
        labels = np.zeros((self.train.shape[0]))
        prediction = np.zeros((self.test.shape[0]))

        clf_list = []

        self.logger.log(logging.DEBUG, "[train cols] " + "-" * 50)
        self.logger.log(logging.DEBUG, model_conf["train_cols"])
        self.validation_scores = []

        for i, (trn_idx, val_idx) in enumerate(sp.idx_list):
            self.logger.log(logging.DEBUG, "-" * 60)
            self.logger.log(logging.DEBUG, f"start training: {i}")

            with timer(f"fold {i}", self.logger):
                _train = self.train.copy()
                if self.another_train:
                    _train = pd.concat([_train, self.another_train])
                    trn_idx = np.concatenate([trn_idx, self.another_train_idx])
                if trn_mode == 'simple':
                    pass
                elif trn_mode == 'last_truncated':
                    trn_idx = self.get_last_trancated_idx(_train, trn_idx)
                if val_mode == 'simple':
                    pass
                elif val_mode == 'last_truncated':
                    val_idx = self.get_last_trancated_idx(_train, val_idx)

                train_df, valid_df = _train.loc[trn_idx], _train.loc[val_idx]

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
                labels[val_idx] = valid_df['accuracy_group'].values

                prediction += fold_prediction / n_splits

                feature_importance_df["fold"] = i
                self.feature_importance.append(feature_importance_df)

#         self.logger.log(logging.DEBUG,
# f"Total Validation Score: {sum(self.validation_scores) /
# len(self.validation_scores):,.5f}")

        self.feature_importance = pd.concat(self.feature_importance, axis=0)

        return clf_list, oof, prediction, self.feature_importance, labels

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

    def get_last_trancated_idx(self, df, idx):
        df = df.loc[idx]
        idx = df\
            .sort_values(['installation_id', 'f019_bef_target_cnt'])\
            .drop_duplicates('installation_id', keep='last').index
        return idx


def exclude(reduce_train, reduce_test, features):
    to_exclude = []
    ajusted_test = reduce_test.copy()
    print('----------------- start excluding features ------------------')
    for feature in features:
        if feature not in ['accuracy_group',
                           'installation_id',
                           'game_session',
                           'title_enc']:
            data = reduce_train[feature]
            train_mean = data.mean()
            data = ajusted_test[feature]
            test_mean = data.mean()
            try:
                ajust_factor = train_mean / test_mean
                if ajust_factor > 10 or ajust_factor < 0.1:  # or error > 0.01:
                    to_exclude.append(feature)
                    print(feature)
                else:
                    ajusted_test[feature] *= ajust_factor
            except BaseException:
                to_exclude.append(feature)
                print(feature)
    print('----------------- done ------------------')
    return to_exclude, ajusted_test
