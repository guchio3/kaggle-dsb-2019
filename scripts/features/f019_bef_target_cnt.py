import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class befTargetCntFeatures(Features):
    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            df = df.loc[df.installation_id.isin(
                self.train_labels.installation_id.unique())]
        else:
            # 直前までのnum_correct/incorrectを取得する
            df = org_test

        ret = applyParallel(
            df.groupby("installation_id"),
            self._calc_features)

        self.format_and_save_feats(ret)
        return ret

    def _calc_features(self, df):
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['test_last'] = 0
        if self.datatype == 'test':
            df.loc[df.shape[0]-1, 'test_last'] = 1

        df = df[
            ((df.type == 'Assessment')
             & (
                ((df.event_code == 4100)
                 & (df.title != 'Bird Measurer (Assessment)'))
                | ((df.event_code == 4110)
                   & (df.title == 'Bird Measurer (Assessment)'))
            ))
            | (df.test_last == 1)
        ]

        res_df = df.groupby(['installation_id', 'game_session']).agg(
            {
                'timestamp': ['max'],
            }
        )
        res_df.columns = [f'{col[0]}_{col[1]}' for col in res_df.columns]
        res_df = res_df.sort_values('timestamp_max')
        res_df['bef_target_cnt'] = res_df.rolling(
            window=len(res_df), min_periods=1).count().values
        res_df = res_df.drop(['timestamp_max'], axis=1)
        res_df = res_df.reset_index()

        if self.datatype == "test":
            res_df = pd.DataFrame([res_df.iloc[-1, :]])
        else:
            # to save memory
            res_df = res_df[
                res_df.game_session
                .isin(self.train_labels.game_session)
            ]

        res_df = res_df\
            .set_index(['installation_id', 'game_session'])\
            .add_prefix(f'{FEATURE_ID}_')\
            .reset_index()

        return res_df
