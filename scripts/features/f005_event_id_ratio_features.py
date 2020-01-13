import gc
import os

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class eventIDRatioFeatures(Features):
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
            org_test.loc[(org_test.event_code.isin([4100, 4110])) & (
                org_test["event_data"].str.contains("true")), 'num_correct'] = 1
            org_test.loc[(org_test.event_code.isin([4100, 4110])) & (
                org_test["event_data"].str.contains("false")), 'num_incorrect'] = 1
            df = org_test

        ret = applyParallel(
            df.groupby("installation_id"),
            self.ins_id_sessions)
        ret_col = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy",
                                                             "game_session", "installation_id", "title",
                                                             "type"
                                                             ]]
        ret[ret_col] = ret[ret_col].fillna(0).astype("int32")
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
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある

        pv = pd.pivot_table(df, index=["installation_id", "gs_max_time", "game_session", "type"],
                            columns="event_id",
                            values="timestamp",
                            aggfunc="count").fillna(0)

        # 時刻順に並ぶことを保証する
        pv.sort_values("gs_max_time", ascending=True, inplace=True)
        pv.reset_index(inplace=True)

        cum_cols = [
            c for c in list(
                pv.columns) if c not in [
                "installation_id",
                "type",
                "game_session",
                "gs_max_time"]]
        pv[cum_cols] = pv[cum_cols].cumsum().shift(1).fillna(0).astype("int32")

        pv = pv.loc[pv["type"] == "Assessment"]  # assessment だけとればOK

        pv_cum_col_sum = pv[cum_cols].sum(axis=1).values.reshape(-1, 1)
        pv[cum_cols] = (pv[cum_cols] / pv_cum_col_sum).values

        rename_dict = {}
        for c in cum_cols:
            rename_dict[c] = f"{FEATURE_ID}_event_id_ratio_" + str(c)
        pv.rename(columns=rename_dict, inplace=True)
        pv.reset_index(inplace=True, drop=True)

        del pv["gs_max_time"], pv["type"]
        gc.collect()

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])

        return pv
