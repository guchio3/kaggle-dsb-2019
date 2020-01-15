import os

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel, groupings

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class immediatelyBeforeFeatures(Features):
    """
    date features

    """

    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def get_encoder(self, org_train, org_test):
        self.all_activities = np.sort(list(set(org_train["title"].unique()).union(
            set(org_test["title"].unique()))))
        self.all_event_codes = np.sort(list(set(org_train["event_code"].unique()).union(
            set(org_test["event_code"].unique()))))
        # self.all_activities = set(org_train["title"].unique()).union(
        #     set(org_test["title"].unique()))
        # self.all_event_codes = set(org_train["event_code"].unique()).union(
        #     org_test["event_code"].unique())
        self.activities_map = dict(
            zip(self.all_activities, np.arange(len(self.all_activities))))
        self.inverse_activities_map = dict(
            zip(np.arange(len(self.all_activities)), self.all_activities))

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            df = df.loc[df.installation_id.isin(
                self.train_labels.installation_id.unique())]
        else:
            # 直前までのnum_correct/incorrectを取得する
            df = org_test

        # get encodings informations
        self.get_encoder(org_train, org_test)

        ret = applyParallel(
            df.groupby("installation_id"),
            self._calc_features)

        self.format_and_save_feats(ret)

        return ret

    def _calc_features(self, df):
        """
        """
        SHIFT = 1

        # game session 毎に feature を作る
        df = df.sort_values(['game_session', 'timestamp'])\
            .reset_index(drop=True)
        grp_features_df = df.groupby(['installation_id', 'game_session']).agg(
            {
                'timestamp': ['max'],
                'event_count': ['max'],
                'event_code': {
                    'last': lambda x: x.iloc[-1]
                },
                'game_time': {
                    'max': 'max',
                    'skew': 'skew',
                    'kurt': lambda x: x.kurt()
                },
                'title': {
                    'LE': lambda x: self.activities_map[x.iloc[-1]],
                },
                'type': {
                    'LE': lambda x: {
                        'Game': 0,
                        'Activity': 1,
                        'Assessment': 2,
                        'Clip': 3,
                    }[x.iloc[-1]]
                },
                'world': {
                    'LE': lambda x: {
                        'MAGMAPEAK': 0,
                        'TREETOPCITY': 1,
                        'CRYSTALCAVES': 2,
                        'NONE': 3,
                    }[x.iloc[-1]]
                },
            }
        )
        # col 調整
        grp_features_df.columns = [
            f'{col[0]}_{col[1]}' for col in grp_features_df.columns]
        # session 順を保証
        grp_features_df = grp_features_df.sort_values('timestamp_max')

        # shift
        res_grp_features_df = grp_features_df\
            .set_index(['installation_id', 'game_session'])\
            .add_prefix(f'{FEATURE_ID}_{SHIFT}th_before_session_')\
            .shift(SHIFT)\
            .reset_index()

        return res_grp_features_df
