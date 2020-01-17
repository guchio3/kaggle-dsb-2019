import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class targetFeatures(Features):
    def __init__(self, params, logger=None):
        super().__init__(params, logger=logger)

    def calc_feature(self, df):
        target_c_ass_idx = ((df.event_code == 4100)
                            & (df.title != "Bird Measurer (Assessment)")
                            & (df.type == "Assessment")
                            & (df["event_data"].str.contains("true"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df.type == "Assessment")
             & (df["event_data"].str.contains("true")))

        target_inc_ass_idx = ((df.event_code == 4100)
                              & (df.title != "Bird Measurer (Assessment)")
                              & (df.type == "Assessment")
                              & (df["event_data"].str.contains("false"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df.type == "Assessment")
             & (df["event_data"].str.contains("false")))

        df.loc[target_c_ass_idx, 'num_correct'] = 1
        df.loc[target_inc_ass_idx, 'num_incorrect'] = 1

        df = df[
            (df.type == 'Assessment')
            & (
                ((df.event_code == 4100)
                 & (df.title != 'Bird Measurer (Assessment)'))
                | ((df.event_code == 4110)
                   & (df.title == 'Bird Measurer (Assessment)'))
            )
        ]
        print(df.shape)

        ret = applyParallel(
            df.groupby("installation_id"),
            self._calc_features)

        self.format_and_save_feats(ret)
        return ret

    def _calc_features(self, df):
        res_df = df.groupby(['installation_id', 'game_session']).agg(
            {
                'num_correct': ['sum'],
                'num_incorrect': ['sum'],
            }
        )
        res_df.columns = [col[0] for col in res_df.columns]
        res_df['accuracy_group'] = res_df.apply(
            self._calc_accuracy_group, axis=1)

        res_df = res_df.reset_index()
        return res_df

    def _calc_accuracy_group(self, row):
        accuracy = row['num_correct'] / \
            (row['num_correct'] + row['num_incorrect'])
        if accuracy == 0:
            return 0
        elif accuracy == 1:
            return 3
        elif accuracy == 0.5:
            return 2
        else:
            return 1
