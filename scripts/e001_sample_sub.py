import importlib
import os

import numpy as np
import pandas as pd

from features.f001_suga_yama_features import (EncodingTitles, EventCount,
                                              EventCount2,
                                              PrevAssessAccByTitle,
                                              SessionTime2, Worldcount)
from yamakawa_san_utils import OptimizedRounder, Validation, log_output, qwk, preprocess_dfs

# set configs
EXP_ID = os.path.basename(__file__).split('_')[0]

CONFIG_MODULE = f'configs.{EXP_ID}_config'
CONFIG_DICT = importlib.import_module(CONFIG_MODULE).CONFIG_DICT


exp_name = "suga_001_add_eventidcnt"
logger, log_path = log_output(exp_name)

train_small_dataset = False
is_debug = False
is_local = False


def main():
    train_df = pd.read_pickle('../mnt/inputs/origin/train.pkl.gz')
    test_df = pd.read_csv('../mnt/inputs/origin/test.csv')

    # train_params = {
    #     'learning_rate': 0.01,
    #     'bagging_fraction': 0.90,
    #     'feature_fraction': 0.85,
    #     'max_depth': 5,
    #     'lambda_l1': 0.7,
    #     'lambda_l2': 0.7,
    #     'metric': 'multiclass',
    #     'objective': 'multiclass',
    #     'num_classes': 4,
    #     'random_state': 773,
    #     "n_estimators": 3000

    # }

    train_params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 64,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'max_depth': -1,
        'lambda_l1': 0.2,
        'lambda_l2': 0.4,
        'seed': 19930802,
        'n_estimators': 100000
    }

    bad_feats = [
        'prev_gs_duration',
        'session_intervalrmin',
        'session_intervalrstd',
        'session_intervalrmax',
        'session_interval',
        'accum_acc_gr_-99',
        'session_intervalrmean',
        'ass_session_interval',
        'prev_gs_durationrmean',
        'prev_gs_durationrmax',
        'ev_cnt4070',
        'prev_gs_durationrstd',
        'mean_g_duration_meaan',
        'ev_cnt3010',
        'g_duration_std',
        'ev_cnt4030',
        'ev_cnt3110',
        'g_duration_mean',
        'meaan_g_duration_min',
        'ass_session_interval_rmin',
        'accum_acc_gr_3',
        'g_duration_min',
        'mean_g_duraation_std'
    ]

    no_use_cols = [
        "accuracy",
        "accuracy_group",
        "game_session",
        "installation_id",
        "title",
        "type",
        "world",
        "pred_y"
    ] + list(set(train_df.columns) - set(test_df.columns)) + bad_feats

    train_cols = [c for c in list(train_df.columns) if c not in no_use_cols]

    print(f"train_df shape: {train_df.shape}")
    print(train_cols)

    cat_cols = [
    ]

    # logger.log(logging.DEBUG, f"categorical cols: {cat_cols}")

    target = "accuracy_group"

    model_conf = {
        "predict_type": "regressor",
        "train_params": train_params,
        "train_cols": train_cols,
        "cat_cols": cat_cols,
        "target": target,
        "is_debug": is_debug,
    }

    validation_param = {
        "model_name": "LGBM",
    }

    exp_conf = {
        "train_small_dataset": False,
        "use_feature": {
            "sample": True
        },
        "train_params": train_params,
        "exp_name": exp_name
    }

    # ==============================
    # start processing
    # ==============================
    use_feature = {
        "EventCount": [EventCount, True],  # class, is_overwrite
        "EventCount2": [EventCount2, True],  # class, is_overwrite
        "Worldcount": [Worldcount, True],
        "SessionTime": [SessionTime2, True],
        #     "AssessEventCount": [AssessEventCount, False],
        "EncodingTitles": [EncodingTitles, True],
        #     "PrevAssessResult":[PrevAssessResult, True],
        #     "PrevAssessAcc": [PrevAssessAcc, True],
        "PrevAssessAccByTitle": [PrevAssessAccByTitle, True]
    }

    is_local = False

    if is_local:
        base_path = "../input"  # at local
        train_df, test_df = preprocess_dfs(
            use_feature, is_local=is_local, logger=None, debug=False)

    else:
        sub = pd.read_csv(
            '../input/data-science-bowl-2019/sample_submission.csv')
        base_path = '/kaggle/input/data-science-bowl-2019'  # at kaggle kernel
#        if len(sub) == 1000:
        if False:
            sub.to_csv('submission.csv', index=False)
            exit(0)
        else:
            train_df, test_df = preprocess_dfs(
                use_feature, is_local=is_local, logger=None, debug=is_debug)

    v = Validation(validation_param, exp_conf, train_df, test_df, logger)
    clf, oof, prediction, feature_importance = v.do_valid_kfold(model_conf)

    test_pred = prediction.copy()

    optR = OptimizedRounder()
    optR.fit(oof, train_df[target])
    coefficients = optR.coefficients()

    opt_preds = optR.predict(oof, coefficients)
    print(qwk(train_df[target], opt_preds))
