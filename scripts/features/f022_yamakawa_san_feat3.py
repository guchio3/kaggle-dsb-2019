import gc
import os
from io import StringIO

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel


class PrevAssessAccByTitle2(Features):
    """kernel features revised
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
            self.ins_id_sessions)
        ret_col = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy",
                                                             "game_session", "installation_id", "title",
                                                             "type"
                                                             ]]
#         self.format_and_save_feats(ret)

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
        # num incorrectを取得
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある

        c_ass_idx = (((df.event_code == 4100)
                      & (df.title != "Bird Measurer (Assessment)")
                      & (df["event_data"].str.contains("true"))) |
                     ((df.event_code == 4110)
                      & (df.title == "Bird Measurer (Assessment)")
                      & (df["event_data"].str.contains("true"))) & (df["type"] == "Assessment"))

        inc_ass_idx = (((df.event_code == 4100)
                        & (df.title != "Bird Measurer (Assessment)")
                        & (df["event_data"].str.contains("false"))) |
                       ((df.event_code == 4110)
                        & (df.title == "Bird Measurer (Assessment)")
                        & (df["event_data"].str.contains("false"))) & (df["type"] == "Assessment"))

        df.loc[c_ass_idx, 'num_correct'] = 1
        df.loc[inc_ass_idx, 'num_incorrect'] = 1

        df["num_correct"] = df["num_correct"].fillna(0)
        df["num_incorrect"] = df["num_incorrect"].fillna(0)

        # ----------------------
        # 全assessmentを取得　
        assess_df = df.loc[(df.type == "Assessment")][["installation_id",
                                                       "game_session",
                                                       "gs_max_time",
                                                       "title"]].drop_duplicates().sort_values("gs_max_time")

        # 少なくとも1つcorrect/incorrectを含むものに限定する
        num_c_df = df.loc[(df.type == "Assessment") &
                          ((df.num_correct > 0) | (df.num_incorrect > 0))].groupby(["game_session"])[["num_correct", "num_incorrect"]].sum().reset_index()

        assess_df = pd.merge(
            assess_df,
            num_c_df,
            how="left",
            on="game_session")

        # test setにおいては、last game-sessionも取る必要がある
        if self.datatype == "test":
            last_session_name = assess_df.sort_values(
                "gs_max_time")["game_session"].values[-1]
        else:
            last_session_name = "train_dummy_session"

        assess_df = assess_df.loc[((assess_df.num_correct > 0) | (assess_df.num_incorrect > 0)) | (
            assess_df.game_session == last_session_name)]  # train labelsのsessionは(pv.num_correct > 0) | (pv.num_incorrect > 0)を常に満たす
        gc.collect()

        titles = assess_df.title.unique()
        assess_df = assess_df.set_index(["installation_id",
                                         "game_session",
                                         "gs_max_time",
                                         "title"]).unstack().sort_values("gs_max_time")
        assess_df = assess_df.shift(1).fillna(0)
        assess_df.columns = [c[0] + "_" + c[1] for c in assess_df.columns]

        # cumsum of num_correct / incorrect
        # --- correct
        num_correct_cols = ["num_correct_" + c for c in titles]
        cumsum_correct_cols = ["cumsum_" + c for c in num_correct_cols]
        assess_df[cumsum_correct_cols] = assess_df[num_correct_cols].cumsum()

        # --- correct
        num_incorrect_cols = ["num_incorrect_" + c for c in titles]
        cumsum_incorrect_cols = ["cumsum_" + c for c in num_incorrect_cols]
        assess_df[cumsum_incorrect_cols] = assess_df[num_incorrect_cols].cumsum()

        # cum accuracy
        for c in titles:
            assess_df["cum_accuracy_" + c] = assess_df["cumsum_num_correct_" + c] / \
                (assess_df["cumsum_num_correct_" + c] +
                 assess_df["cumsum_num_incorrect_" + c])

        assess_df = assess_df.reset_index().sort_values("gs_max_time")

        del assess_df["gs_max_time"]

        if self.datatype == "test":
            assess_df = pd.DataFrame([assess_df.iloc[-1, :]])

        return assess_df
