import os

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
        """
        """
        df["gs_max_time"] = df.groupby("game_session")["timestamp"]\
            .transform("max")  # gs_max_timeでsortする必要がある
        df = df.sort_values('gs_max_time')

        # 計算量削減のため、必要なもののみ取得
        # この class の場合、ass とその直前の session のみ必要
        ass_idx = (((df.event_code == 4100)
                    & (df.type == "Assessment")
                    & (df.title != "Bird Measurer (Assessment)"))
                   | ((df.event_code == 4110)
                      & (df.type == "Assessment")
                      & (df.title == "Bird Measurer (Assessment)"))).values
        ass_sessions = df[ass_idx].game_session.unique()
        df['is_in_ass_session'] = 

        df.groupby()

        # 計算量削減のため、対象行に限定
        df = df[ass_idx]

        df = df.set_index(['game_session', 'installation_id'])\
            .add_prefix(f'{FEATURE_ID}_dt_')\
            .reset_index()
        return df

    def _
