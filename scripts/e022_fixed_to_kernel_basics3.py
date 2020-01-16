# import importlib
import logging
import os
import pickle
from pathlib import Path

# import numpy as np
import numpy as np
import pandas as pd

from features.f001_suga_yama_features import (EncodingTitles, EventCount,
                                              EventCount2, KernelBasics2,
                                              PrevAssessAccByTitle,
                                              SessionTime2, Worldcount)
from features.f002_dt_features import dtFeatures
from features.f004_event_code_ratio_features import eventCodeRatioFeatures
from features.f005_event_id_ratio_features import eventIDRatioFeatures
from features.f007_encoding_title_order import encodingTitleOrder
from features.f008_immediately_before_features import immediatelyBeforeFeatures
from features.f009_world_label_encoding_diff import worldLabelEncodingDiffFeatures
from features.f010_world_numerical_features import worldNumeriacalFeatures
from features.f011_world_assessment_numerical_features import worldAssessmentNumeriacalFeatures
from features.f999_suga_yama_features_fixed import KernelBasics3
# from guchio_utils import guchioValidation
from yamakawa_san_utils import (OptimizedRounder, Validation, base_path,
                                log_output, memory_reducer, pickle_load, qwk,
                                timer)

# set configs
EXP_ID = os.path.basename(__file__).split('_')[0]

# CONFIG_MODULE = f'configs.{EXP_ID}_config'
# CONFIG_DICT = importlib.import_module(CONFIG_MODULE).CONFIG_DICT


exp_name = EXP_ID
logger, log_path = log_output(exp_name)

train_small_dataset = False
is_debug = False
is_local = False


def feature_maker(feat_cls, is_overwrite, org_train, org_test,
                  train_labels, params, logger, is_local):
    """featureの読み込み
    """
    feat_ = feat_cls(train_labels, params, logger)
    feat_name = feat_.name
    datatype = feat_.datatype
    feature_dir = './mnt/inputs/features'
    # feature_dir = os.path.join(os.path.dirname("__file__"), "../feature")
    feature_path = Path(feature_dir) / f"{datatype}" / f"{feat_name}.pkl"

    print(f'feature_path: {feature_path}')
    print(f'os.path.exists(feature_path): {os.path.exists(feature_path)}')
    print(f'is_overwrite: {is_overwrite}')
    if os.path.exists(feature_path) and is_overwrite is False:
        f_df = pickle_load(feature_path)
    else:
        f_df = feat_.feature_extract(org_train, org_test)

    return f_df


def add_features(use_features, org_train, org_test, train_labels,
                 specs, datatype, is_local=False, logger=None):
    # 都度計算する
    feat_params = {
        "datatype": datatype,
        "debug": True,
        "is_overwrite": True,
    }

    # base feature
    base_feat = KernelBasics3(train_labels, feat_params, logger)
    feature_dir = './mnt/inputs/features'
    # feature_dir = os.path.join(os.path.dirname("__file__"), "../feature")
    feature_path = Path(feature_dir) / f"{datatype}" / f"{base_feat.name}.pkl"

    if os.path.exists(feature_path):
        feat_df = pickle_load(feature_path)
    else:
        feat_df = base_feat.feature_extract(org_train, org_test)

    # add event_counts
    for name, feat_condition in use_features.items():
        feat_cls = feat_condition[0]
        is_overwrite = feat_condition[1]

        f_df = feature_maker(
            feat_cls,
            is_overwrite,
            org_train,
            org_test,
            train_labels,
            feat_params,
            logger,
            is_local)
        feat_df = pd.merge(
            feat_df, f_df, how="left", on=[
                "installation_id", "game_session"])
        del f_df

    return feat_df


def preprocess_dfs(use_features, is_local=False, logger=None, debug=True):
    # read dataframes
    with timer("read datasets"):
        if debug:
            nrows = 200000
        else:
            nrows = None

        sub = pd.read_csv(base_path + '/sample_submission.csv')

        if is_local:
            org_train = pickle_load("../input/train.pkl")
            org_test = pickle_load("../input/test.pkl")
        else:
            org_train = pd.read_csv(base_path + "/train.csv", nrows=nrows)
            org_test = pd.read_csv(base_path + "/test.csv", nrows=nrows)

        org_train = memory_reducer(org_train, verbose=True)
        org_test = org_test[org_test.installation_id.isin(sub.installation_id)]
        org_test.sort_values(['installation_id', 'timestamp'], inplace=True)
        org_test.reset_index(inplace=True)
        org_test = memory_reducer(org_test, verbose=True)

        train_labels = pd.read_csv(
            base_path + "/train_labels.csv", nrows=nrows)
        specs = pd.read_csv(base_path + "/specs.csv", nrows=nrows)

    # basic preprocess
    org_train["timestamp"] = pd.to_datetime(org_train["timestamp"])
    org_test["timestamp"] = pd.to_datetime(org_test["timestamp"])

    with timer("merging features"):
        train_df = add_features(
            use_features,
            org_train,
            org_test,
            train_labels,
            specs,
            datatype="train",
            is_local=is_local,
            logger=None)
        train_df = train_df.reset_index(drop=True)
        test_df = add_features(
            use_features,
            org_train,
            org_test,
            train_labels,
            specs,
            datatype="test",
            is_local=is_local,
            logger=None)
        test_df = test_df.reset_index(drop=True)

#     df = pd.concat([df, feat_df], axis=1)
    print("preprocess done!!")

    return train_df, test_df


def main():
    train_df = pd.read_pickle('./mnt/inputs/origin/train.pkl.gz')
    test_df = pd.read_csv('./mnt/inputs/origin/test.csv')

    # ==============================
    # start processing
    # ==============================
    use_feature = {
        "EventCount": [EventCount, False],  # class, is_overwrite
        "EventCount2": [EventCount2, False],  # class, is_overwrite
        "Worldcount": [Worldcount, False],
        "SessionTime": [SessionTime2, False],
        #     "AssessEventCount": [AssessEventCount, False],
        "EncodingTitles": [EncodingTitles, False],
        # "encodingTitleOrder": [encodingTitleOrder, False],
        #     "PrevAssessResult":[PrevAssessResult, True],
        #     "PrevAssessAcc": [PrevAssessAcc, True],
        "PrevAssessAccByTitle": [PrevAssessAccByTitle, False],
        "dtFeatures": [dtFeatures, False],
        # "eventCodeRatioFeatures": [eventCodeRatioFeatures, False],
        # "eventIDRatioFeatures": [eventIDRatioFeatures, False],
        "immediatelyBeforeFeatures": [immediatelyBeforeFeatures, False],
        # "worldLabelEncodingDiffFeatures": [worldLabelEncodingDiffFeatures, False],
        "worldNumeriacalFeatures": [worldNumeriacalFeatures, False],
    }

    is_local = False

    if is_local:
        base_path = "../input"  # at local
        train_df, test_df = preprocess_dfs(
            use_feature, is_local=is_local, logger=None, debug=False)

    else:
        base_path = './mnt/inputs/origin'  # at kaggle kernel
        sub = pd.read_csv(
            f'{base_path}/sample_submission.csv')
#        base_path = '/kaggle/input/data-science-bowl-2019'  # at kaggle kernel
#        if len(sub) == 1000:
        if False:
            sub.to_csv('submission.csv', index=False)
            exit(0)
        else:
            train_df, test_df = preprocess_dfs(
                use_feature, is_local=is_local, logger=None, debug=is_debug)

    # remove , to avoid error of lgbm
    train_df.columns = [col.replace(',', '_') for col in train_df.columns]
    test_df.columns = [col.replace(',', '_') for col in test_df.columns]

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
        # 'num_leaves': 16,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'max_depth': -1,
        'lambda_l1': 0.2,
        'lambda_l2': 0.4,
        'seed': 19930802,
        'n_estimators': 100000,
        'importance_type': 'gain',
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
    # target = "accuracy"

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

    # train_df[target] = train_df[target] / 3.
    # train_df.loc[train_df[target] <= 1, target] = 0
    # train_df.loc[train_df[target] > 1, target] = 1
    # print(train_df[target].head())
    v = Validation(validation_param, exp_conf, train_df, test_df, logger)
    clf, oof, prediction, feature_importance_df \
        = v.do_valid_kfold(model_conf)

    optR = OptimizedRounder()
    optR.fit(oof, train_df['accuracy_group'])
    coefficients = optR.coefficients()

    opt_preds = optR.predict(oof, coefficients)

    oof_dir = f'./mnt/oofs/{EXP_ID}'
    if not os.path.exists(oof_dir):
        os.mkdir(oof_dir)
    with open(f'{oof_dir}/{EXP_ID}_oof.pkl', 'wb') as fout:
        pickle.dump(oof, fout)

    res_qwk = qwk(train_df['accuracy_group'], opt_preds)
    print(f'res_qwk : {res_qwk}')
    logger.log(
        logging.DEBUG,
        f'qwk -- {res_qwk}'
    )

#     print(f'qwk -- {np.mean(valid_qwks)} +- {np.std(valid_qwks)}')
#     logger.log(
#         logging.DEBUG,
#         f'qwk -- {np.mean(valid_qwks)} +- {np.std(valid_qwks)}')

    # save info
    feature_importance_df.to_csv(
        f'./mnt/importances/{EXP_ID}.csv', index=False)


if __name__ == '__main__':
    main()
