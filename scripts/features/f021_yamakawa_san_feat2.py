import gc
import os
from io import StringIO

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel


class UserActivityCount(Features):
    """kernel features revised
    """

    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            assess_user = df.loc[df.type ==
                                 "Assessment"].installation_id.unique()
            df = df.loc[df.installation_id.isin(assess_user)]
        else:
            # 直前までのnum_correct/incorrectを取得する
            df = org_test

        ret = applyParallel(
            df.groupby("installation_id"),
            self.ua_count_sessions)
        ret_col = [
            c for c in list(
                ret.columns) if c not in [
                "game_session",
                "installation_id",
                "title",
                "type"]]
        ret[ret_col] = ret[ret_col].fillna(0).astype("int32")

        use_cols = [
            c for c in list(
                ret.columns) if c not in [
                "title",
                "type",
                "event_code",
                "gs_max_time"]]
        self.format_and_save_feats(ret[use_cols])
        return ret[use_cols]

    def ua_count_sessions(self, df):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        df["gs_min_time"] = df.groupby("game_session")["timestamp"].transform(
            "min")  # gs_max_timeでsortする必要がある

        pv = pd.pivot_table(df, index=["installation_id", "gs_min_time", "game_session"],
                            columns="type",
                            values="timestamp",
                            aggfunc="count").fillna(0)

        pv = np.sign(pv).sort_values("gs_min_time")  # add flag
        pv = pv.cumsum().shift(1).reset_index()

        gc.collect()
        pv = pv.loc[pv.game_session.isin(
            df.loc[df.type == "Assessment"].game_session.unique())].sort_values("gs_min_time")
        del pv["gs_min_time"]

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])

        return pv
