import gc
import os

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class EncodingTitles(Features):
    """Event count in only Assessments
    """

    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def get_encoder(self, org_train, org_test):
        self.all_activities = np.sort(list(set(org_train["title"].unique()).union(
            set(org_test["title"].unique()))))
        # self.all_activities = set(org_train["title"].unique()).union(
        #     set(org_test["title"].unique()))
        self.all_event_codes = set(org_train["event_code"].unique()).union(
            org_test["event_code"].unique())
        self.activities_map = dict(
            zip(self.all_activities, np.arange(len(self.all_activities))))
        self.inverse_activities_map = dict(
            zip(np.arange(len(self.all_activities)), self.all_activities))

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            assess_user = df.loc[df.type ==
                                 "Assessment"].installation_id.unique()
            df = df.loc[df.installation_id.isin(assess_user)]
        else:
            # 直前までのnum_correct/incorrectを取得する
            org_test.loc[(org_test.event_code.isin([4100, 4110])) & (
                org_test["event_data"].str.contains("true")), 'num_correct'] = 1
            org_test.loc[(org_test.event_code.isin([4100, 4110])) & (
                org_test["event_data"].str.contains("false")), 'num_incorrect'] = 1
            df = org_test

        # get encodings informations
        self.get_encoder(org_train, org_test)

        ret = applyParallel(
            df.groupby("installation_id"),
            self.ins_id_sessions)
        use_cols = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy", "title",
                                                              "type", "event_code", "gs_max_time"
                                                              ]]

        self.format_and_save_feats(ret[use_cols])
        return ret[use_cols]

    def ins_id_sessions(self, df):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        df["title_enc"] = df["title"].map(self.activities_map)
        df = df.loc[df.type == "Assessment"][["installation_id",
                                              "game_session",
                                              "title_enc"]].drop_duplicates().reset_index(drop=True)

        if self.datatype == "test":
            df = pd.DataFrame([df.iloc[-1, :]])

        return df
