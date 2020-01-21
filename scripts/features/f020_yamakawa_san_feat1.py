import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel


class TypeEventCounts(Features):
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
            self.type_event_count)
        ret_col = [
            c for c in list(
                ret.columns) if c not in [
                "game_session",
                "installation_id",
                "title",
                "type"]]

        use_cols = [
            c for c in list(
                ret.columns) if c not in [
                "title",
                "type",
                "event_code",
                "gs_max_time"]]
        self.format_and_save_feats(ret[use_cols])

        return ret[use_cols]

    def type_event_count(self, df):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        df["gs_min_time"] = df.groupby("game_session")["timestamp"].transform(
            "min")  # gs_max_timeでsortする必要がある

        agg_dict = {
            "event_count": ["min", "max"]
        }
        # groupings(df, ["installation_id", "game_session", "gs_min_time", "type"], agg_dict, pref="event_count")
        type_max_event_df = df.groupby(["installation_id", "game_session", "gs_min_time", "type"])[
            "event_count"].max().unstack().sort_values("gs_min_time", ascending=True)

        type_max_event_df = type_max_event_df.shift(1)

        types = list(type_max_event_df.columns)
        type_means = ["mean_" + c for c in types]
        type_max_event_df[type_means] = type_max_event_df[types].rolling(
            window=len(type_max_event_df), min_periods=1).mean()
        type_max_event_df = type_max_event_df.reset_index()

        rename_dict = {c: "prev_max_ev_cnt_" + c for c in types}
        type_max_event_df.rename(columns=rename_dict, inplace=True)

        assess_ids = df.loc[df.type == "Assessment"]["game_session"].unique()
        type_max_event_df = type_max_event_df.loc[type_max_event_df.game_session.isin(
            assess_ids)].sort_values("gs_min_time")

        del assess_ids

        gc.collect()
        del type_max_event_df["gs_min_time"]

        if self.datatype == "test":
            type_max_event_df = pd.DataFrame([type_max_event_df.iloc[-1, :]])

        return type_max_event_df
