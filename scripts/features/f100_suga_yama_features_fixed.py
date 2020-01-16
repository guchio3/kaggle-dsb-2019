import gc
import re

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel, groupings


class KernelBasics3(Features):
    """kernel features revised
    """

    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def mapping_codes(self, org_train, org_test):
        self.all_activities = set(org_train["title"].unique()).union(
            set(org_test["title"].unique()))
        self.all_event_codes = set(org_train["event_code"].unique()).union(
            org_test["event_code"].unique())

        # convert activities <=> int
        self.activities_map = dict(
            zip(self.all_activities, np.arange(len(self.all_activities))))  # activity title => int
        self.inverse_activities_map = dict(
            zip(np.arange(len(self.all_activities)), self.all_activities))  # int => activity title

        # convert win_code <=> int
        win_code = dict(
            zip(activities_map.values(),
                (4100 * np.ones(len(activities_map))).astype(int)))
        win_code[activities_map["Bird Measurer (Assessment)"]] = 4110

        self.win_code = win_code

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            df = df.loc[df.installation_id.isin(
                self.train_labels.installation_id.unique())]
        else:
            df = org_test

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

        ret = applyParallel(
            df.groupby("installation_id"),
            self.ins_id_sessions)
        ret_col = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy",
                                                             "game_session", "installation_id", "title",
                                                             "type"
                                                             ]]

        use_cols = [c for c in list(ret.columns) if "Assessment" not in c]
        del ret["accum_acc_gr_-99"], ret["prev_acc_gr_-99"]

        fill_cols = [c for c in list(ret.columns) if c not in ["cum_accuracy", "cum_accuracy", "prev_acc_gr_0", "prev_acc_gr_1",
                                                               "prev_acc_gr_2", "prev_acc_gr_3", "prev_num_corrects", "prev_num_incorrects"]]
        ret[fill_cols] = ret[fill_cols].fillna(0)
        self.format_and_save_feats(ret)

        return ret

    def ins_id_sessions(self, df):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        # initialize user activity
        # 1. time spent
        # 2. event count
        # 3. session count

        # sessionごとのplaytimeを算出
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある
        pv = pd.pivot_table(df, index=["installation_id", "gs_max_time", "game_session", "type"],
                            columns="title",
                            values="game_time",
                            aggfunc="max").fillna(0)

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
#         pv[cum_cols] = (pv[cum_cols].cumsum() // 1000).astype("int32")
        pv[cum_cols] = (pv[cum_cols].cumsum()).fillna(0)
        pv[cum_cols] = pv[cum_cols].shift(1)  # 直前までのplaytimeを取得する

        ins_id = df.installation_id.values[0]
        # calc num corrects
        pv = pv.loc[pv.type == "Assessment"]
        num_c_df = df.loc[(df.type == "Assessment") &
                          ((df.num_correct > 0) | (df.num_incorrect > 0))].groupby(["game_session"])[["num_correct", "num_incorrect"]].sum().reset_index()
        pv = pd.merge(pv, num_c_df, how="left", on="game_session")

        if self.datatype == "test":
            last_session_name = pv.sort_values(
                "gs_max_time")["game_session"].values[-1]
        else:
            last_session_name = "train_dummy_session"

        # train labelsのsessionは(pv.num_correct > 0) | (pv.num_incorrect >
        # 0)を常に満たす
        pv = pv.loc[((pv.num_correct > 0) | (pv.num_incorrect > 0))
                    | (pv.game_session == last_session_name)]
        gc.collect()

        # 直前までの正解状況を集計
        pv["prev_num_corrects"] = pv["num_correct"].shift(1)
        pv["prev_num_incorrects"] = pv["num_incorrect"].shift(1)
        pv["prev_cumnum_c"] = pv["prev_num_corrects"].cumsum()
        pv["prev_cumnum_inc"] = pv["prev_num_incorrects"].cumsum()

        pv["cum_accuracy"] = (pv["prev_cumnum_c"] /
                              (pv["prev_cumnum_c"] + pv["prev_cumnum_inc"]))

        del pv["num_correct"], pv["num_incorrect"]
        gc.collect()

        pv = self.get_acc_group(pv)

        pv = pv.sort_values("gs_max_time").reset_index(drop=True)
        del pv["gs_max_time"]

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])
        return pv

    def get_acc_group(self, pv):
        def calc_accuracy_group(row):
            if row["prev_num_incorrects"] + row["prev_num_corrects"] > 0:
                acc = row["prev_num_corrects"] / \
                    (row["prev_num_incorrects"] + row["prev_num_corrects"])
                if acc == 0:
                    return 0
                elif acc == 1:
                    return 3
                elif acc == 0.5:
                    return 2
                else:
                    return 1
            else:
                return -99

        pv["acc_group"] = pv.apply(calc_accuracy_group, axis=1)
        acc_pv = pd.pivot_table(pv[["gs_max_time",
                                    "installation_id",
                                    "game_session",
                                    "acc_group"]],
                                index=["gs_max_time",
                                       "game_session"],
                                columns="acc_group",
                                values="installation_id",
                                aggfunc="count").reset_index().fillna(0)

        del pv["acc_group"]

        acc_columns = {}
        for col in acc_pv.columns:
            if col in [-99, 0, 1, 2, 3]:
                acc_columns[col] = "prev_acc_gr_" + str(col)
                acc_pv[f"accum_acc_gr_{col}"] = acc_pv[col].cumsum()

        acc_pv.rename(columns=acc_columns, inplace=True)
        del acc_pv["gs_max_time"]
        pv = pd.merge(pv, acc_pv, on="game_session", how="left")

        return pv

    def test_calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            df = df.loc[df.installation_id.isin(
                self.train_labels.installation_id.unique())]
        else:
            df = org_test
#             df = org_train
#             df = df.loc[df.installation_id.isin(self.train_labels.installation_id.unique())]

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

        ins_df = df.loc[df.installation_id == "01a44906"]
        pv = self.ins_id_sessions(ins_df)

        return pv
