import gc
import json
import re

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel, groupings


class KernelBasics2(Features):
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
            # 直前までのnum_correct/incorrectを取得する
            df = org_test
            c_ass_idx = ((df.event_code == 4100)
                         & (df.title != "Bird Measurer (Assessment)")
                         & (df["event_data"].str.contains("true"))) | \
                ((df.event_code == 4110)
                 & (df.title == "Bird Measurer (Assessment)")
                 & (df["event_data"].str.contains("true")))

            inc_ass_idx = ((df.event_code == 4100)
                           & (df.title != "Bird Measurer (Assessment)")
                           & (df["event_data"].str.contains("false"))) | \
                ((df.event_code == 4110)
                 & (df.title == "Bird Measurer (Assessment)")
                 & (df["event_data"].str.contains("false")))

            df.loc[c_ass_idx, 'num_correct'] = 1
            df.loc[inc_ass_idx, 'num_incorrect'] = 1

        ret = applyParallel(
            df.groupby("installation_id"),
            self.ins_id_sessions)
        ret_col = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy",
                                                             "game_session", "installation_id", "title",
                                                             "type"
                                                             ]]
        ret[ret_col] = ret[ret_col].fillna(0).astype("int32")

        use_cols = [c for c in list(ret.columns) if "Assessment" not in c]
        ret = ret[use_cols]

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

        pv = pd.pivot_table(df, index=["gs_max_time", "game_session", "type"],
                            columns="title",
                            values="game_time",
                            aggfunc="max").fillna(0)

        # 時刻順に並ぶことを保証する
        pv.sort_values("gs_max_time", ascending=True, inplace=True)
        pv.reset_index(inplace=True)

        cum_cols = [
            c for c in list(
                pv.columns) if c not in [
                "type",
                "game_session",
                "gs_max_time"]]
        pv[cum_cols] = (pv[cum_cols].cumsum() // 1000).astype("int32")
        pv[cum_cols] = pv[cum_cols].shift(1)  # 直前までのplaytimeを取得する

        ins_id = df.installation_id.values[0]

        # assessmentのrowに限定して抽出する
        if self.datatype == "train":
            # 正解ラベル/num_corrects を得るためtrain labelsとmerge
            pv = pd.merge(pv,
                          self.train_labels[self.train_labels.installation_id == ins_id],
                          how="inner",
                          on="game_session")
        else:
            # calc num corrects
            num_c_df = df.loc[df.type == "Assessment"].groupby(["installation_id", "game_session"])[
                ["num_correct", "num_incorrect"]].sum().fillna(0).reset_index()
            pv = pd.merge(pv, num_c_df, how="left", on="game_session")

        gc.collect()

        # 直前までの正解状況を集計
        pv["prev_num_corrects"] = pv["num_correct"].shift(1).fillna(0)
        pv["prev_cumnum_c"] = pv["prev_num_corrects"].cumsum()
        pv["prev_num_incorrects"] = pv["num_incorrect"].shift(1).fillna(0)
        pv["prev_cumnum_inc"] = pv["prev_num_incorrects"].cumsum()

        pv["cum_accuracy"] = (pv["prev_cumnum_c"] /
                              (pv["prev_cumnum_c"] + pv["prev_cumnum_inc"])).fillna(0)

        del pv["num_correct"], pv["num_incorrect"]
        gc.collect()

        pv = self.get_acc_group(pv)
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


class EventCount(Features):
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
                            columns="event_code",
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

        rename_dict = {}
        for c in cum_cols:
            rename_dict[c] = "ev_cnt" + str(c)
        pv.rename(columns=rename_dict, inplace=True)
        pv.reset_index(inplace=True, drop=True)

        del pv["gs_max_time"], pv["type"]
        gc.collect()

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])

        return pv


class EventCount2(Features):
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

        rename_dict = {}
        for c in cum_cols:
            rename_dict[c] = "ev_cnt" + str(c)
        pv.rename(columns=rename_dict, inplace=True)
        pv.reset_index(inplace=True, drop=True)

        del pv["gs_max_time"], pv["type"]
        gc.collect()

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])

        return pv


class Worldcount(Features):
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

        ret = applyParallel(df.groupby("installation_id"), self.count_sessions)
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

    def count_sessions(self, df):
        world_cnt = self.sub_count_sessions(df, ["world"], "wrd_cnt_")
        world_type_cnt = self.sub_count_sessions(
            df, ["world", "type"], "wrd_type_cnt_")
        title_type_cnt = self.sub_count_sessions(
            df, ["title", "type"], "title_type_cnt_")

        world_cnt = pd.merge(
            world_cnt, world_type_cnt, how="left", on=[
                "installation_id", "game_session"])
        del world_type_cnt

        world_cnt = pd.merge(
            world_cnt, title_type_cnt, how="left", on=[
                "installation_id", "game_session"])
        del title_type_cnt

        return world_cnt

    def sub_count_sessions(self, df, group_columns, prefix):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        assess_sessions = df[df.type == "Assessment"]["game_session"].unique()
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある

        pv = pd.pivot_table(df, index=["installation_id", "gs_max_time", "game_session"],
                            columns=group_columns,
                            values="timestamp",
                            aggfunc="count").fillna(0)

        # 時刻順に並ぶことを保証する
        pv.sort_values("gs_max_time", ascending=True, inplace=True)
        pv.reset_index(inplace=True)

        if len(group_columns) >= 2:
            pv.columns = [c[0] + "_" + c[1] for c in pv.columns]
            pv.rename(columns={"installation_id_": "installation_id",
                               "game_session_": "game_session",
                               "gs_max_time_": "gs_max_time"
                               }, inplace=True)

        cum_cols = [
            c for c in list(
                pv.columns) if c not in [
                "installation_id",
                "game_session",
                "gs_max_time"]]
        pv[cum_cols] = pv[cum_cols].cumsum().shift(1).fillna(0).astype("int32")

        rename_dict = {}
        for c in cum_cols:
            rename_dict[c] = prefix + str(c)
        pv.rename(columns=rename_dict, inplace=True)
        pv.reset_index(inplace=True, drop=True)

        del pv["gs_max_time"]
        gc.collect()

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])
        elif self.datatype == "train":
            pv = pv.loc[pv.game_session.isin(assess_sessions)]

        return pv


class SessionTime2(Features):
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

        ret = applyParallel(df.groupby("installation_id"), self.time_sessions)
        use_cols = [c for c in list(ret.columns) if c not in ["title", "type", "world",
                                                              "event_code", "gs_max_time", "timestamp_max", "timestamp_min"]]
        self.format_and_save_feats(ret[use_cols])

        return ret[use_cols]

    def time_sessions(self, ins_df):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        # session feature for all "type"
        agg_dict = {
            "timestamp": ["max", "min"]
        }
        duration_df = groupings(
            ins_df, [
                "installation_id", "world", "type", "game_session"], agg_dict).sort_values(
            "timestamp_min", ascending=True).reset_index(
                drop=True)

        duration_df["prev_gs_duration"] = (
            duration_df["timestamp_max"] -
            duration_df["timestamp_min"]).shift(1).dt.total_seconds()
        duration_df["session_interval"] = (
            duration_df["timestamp_min"] -
            duration_df["timestamp_max"].shift(1)).dt.total_seconds()

        window = 5
        min_periods = 2
        for col in ["prev_gs_duration", "session_interval"]:
            duration_df[col + "rmean"] = duration_df[col].rolling(
                window=window, min_periods=min_periods).mean()
            duration_df[col + "rstd"] = duration_df[col].rolling(
                window=window, min_periods=min_periods).std()
            duration_df[col + "rmax"] = duration_df[col].rolling(
                window=window, min_periods=2).max()
            duration_df[col + "rmin"] = duration_df[col].rolling(
                window=window, min_periods=2).min()

        duration_df = duration_df.loc[duration_df.type == "Assessment"]

        # session feature for "assessments"
        agg_dict = {
            "timestamp": ["max", "min"]
        }
        ass_duration = groupings(
            ins_df, [
                "installation_id", "world", "type", "game_session"], agg_dict)
        ass_duration = ass_duration.loc[ass_duration.type == "Assessment"].sort_values(
            "timestamp_min", ascending=True).reset_index(drop=True)

        ass_duration["prev_ass_gs_duration"] = (
            ass_duration["timestamp_max"] -
            ass_duration["timestamp_min"]).shift(1).dt.total_seconds()
        ass_duration["ass_session_interval"] = (
            ass_duration["timestamp_min"] -
            ass_duration["timestamp_max"].shift(1)).dt.total_seconds()

        window = 5
        min_periods = 1
        for col in ["prev_ass_gs_duration", "ass_session_interval"]:
            ass_duration[col + "_rmean"] = ass_duration[col].rolling(
                window=window, min_periods=min_periods).mean()
            ass_duration[col + "_rstd"] = ass_duration[col].rolling(
                window=window, min_periods=min_periods).std()
            ass_duration[col + "_rmax"] = ass_duration[col].rolling(
                window=window, min_periods=1).max()
            ass_duration[col + "_rmin"] = ass_duration[col].rolling(
                window=window, min_periods=1).min()

        ass_cols = [
            c for c in list(
                ass_duration.columns) if c not in [
                'installation_id',
                'world',
                'type',
                'timestamp_max',
                'timestamp_min']]

        duration_df = pd.merge(
            duration_df,
            ass_duration[ass_cols],
            how="left",
            on="game_session")

        if self.datatype == "test":
            duration_df = pd.DataFrame([duration_df.iloc[-1, :]])

        return duration_df


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


class PrevAssessAccByTitle(Features):
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
        ret[ret_col] = ret[ret_col].fillna(0)
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
        # 単純なactivity count
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある

        c_ass_idx = ((df.event_code == 4100)
                     & (df.title != "Bird Measurer (Assessment)")
                     & (df["event_data"].str.contains("true"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df["event_data"].str.contains("true")))

        inc_ass_idx = ((df.event_code == 4100)
                       & (df.title != "Bird Measurer (Assessment)")
                       & (df["event_data"].str.contains("false"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df["event_data"].str.contains("false")))

        df.loc[c_ass_idx, 'num_correct'] = 1
        df.loc[inc_ass_idx, 'num_incorrect'] = 1

        df["num_correct"].fillna(0, inplace=True)
        df["num_incorrect"].fillna(0, inplace=True)

        df = df.loc[(df.type == "Assessment")]

        pv = pd.pivot_table(
            df,
            index=[
                "installation_id",
                "game_session",
                "gs_max_time"],
            columns="title",
            values=[
                "num_correct",
                "num_incorrect"],
            aggfunc="sum").reset_index().sort_values("gs_max_time")

        pv.columns = [c[0] + "_" + c[1] if c[1] != "" else c[0]
                      for c in list(pv.columns)]
        pv_num_cols = [c for c in list(pv.columns) if "correct" in c]

        cum_cols = ["cum_" + c for c in pv_num_cols]  # 累積列

        pv_cum_corr_cols = [
            c for c in cum_cols if "cum_num_correct_" in c]  # correct 列のみ
        pv_cum_incorr_cols = [
            c for c in cum_cols if "cum_num_incorrect_" in c]  # incorrect 列
        pv_cum_acc_cols = [
            "cum_acc_" +
            re.sub(
                'num_incorrect_',
                '',
                c) for c in pv_cum_incorr_cols]

        pv.reset_index(drop=True, inplace=True)

        pv[pv_num_cols] = pv[pv_num_cols].shift(1).fillna(0)
        pv[cum_cols] = pv[pv_num_cols].cumsum()

        pv[pv_cum_acc_cols] = pd.DataFrame(
            pv[pv_cum_corr_cols].values / (pv[pv_cum_corr_cols].values + pv[pv_cum_incorr_cols].values))

        del pv["gs_max_time"]

        if self.datatype == "test":
            pv = pd.DataFrame([pv.iloc[-1, :]])

        return pv


class PrevAssessResult(Features):
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
        # 単純なactivity count
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある
        pv = pd.pivot_table(df, index=["installation_id", "gs_max_time", "game_session", "type"],
                            columns="title",
                            values="timestamp",
                            aggfunc="count").fillna(0)

        assess_col = [c for c in list(pv.columns) if "Assessment" in c]
        pv = pv[assess_col]
        pv.reset_index(inplace=True)

        rename_dict = {}
        new_cols = []

        cnt_pref = "assess_cnt_"
        for c in assess_col:
            rename_dict[c] = cnt_pref + str(c)

        pv = pv.loc[pv.type == "Assessment"].reset_index(drop=True)
        pv.sort_values("gs_max_time", ascending=True, inplace=True)
        pv.reset_index(inplace=True, drop=True)
        pv[assess_col] = pv[assess_col].shift(1).fillna(0)
        pv.rename(columns=rename_dict, inplace=True)

        for c in assess_col:
            pv["accum" + cnt_pref + str(c)] = pv[cnt_pref + str(c)].cumsum()

        del pv["gs_max_time"], pv["type"]

        return pv


def assess_history(gr_df):
    gr_df = gr_df.sort_values("gs_max_time", ascending=True)

    gr_df["as_acc_c_num"] = gr_df["num_correct"].cumsum()
    gr_df["as_acc_inc_num"] = gr_df["num_incorrect"].cumsum()
    gr_df["as_prev_acc"] = gr_df["num_correct"] / \
        (gr_df["num_correct"] + gr_df["num_incorrect"])
    gr_df["as_cum_acc"] = gr_df["as_acc_c_num"] / \
        (gr_df["as_acc_c_num"] + gr_df["as_acc_inc_num"])

    shift_col = [
        "num_correct",
        "num_incorrect",
        "as_acc_c_num",
        "as_acc_inc_num",
        "as_prev_acc",
        "as_cum_acc"]
    gr_df[shift_col] = gr_df[shift_col].shift(1).fillna(-99)

    return gr_df


class PrevAssessAcc(Features):
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
            self.ins_id_sessions)
        ret_col = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy",
                                                             "game_session", "installation_id", "title",
                                                             "type"
                                                             ]]
        ret[ret_col] = ret[ret_col].fillna(0)
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
        # 単純なactivity count
        df["gs_max_time"] = df.groupby("game_session")["timestamp"].transform(
            "max")  # gs_max_timeでsortする必要がある

        c_ass_idx = ((df.event_code == 4100)
                     & (df.title != "Bird Measurer (Assessment)")
                     & (df["event_data"].str.contains("true"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df["event_data"].str.contains("true")))

        inc_ass_idx = ((df.event_code == 4100)
                       & (df.title != "Bird Measurer (Assessment)")
                       & (df["event_data"].str.contains("false"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df["event_data"].str.contains("false")))

        df.loc[c_ass_idx, 'num_correct'] = 1
        df.loc[inc_ass_idx, 'num_incorrect'] = 1

        df["num_correct"].fillna(0, inplace=True)
        df["num_incorrect"].fillna(0, inplace=True)

        df = df.loc[(df.type == "Assessment")]

        df = df.groupby(["installation_id", "game_session", "gs_max_time", "title"])[
            ["num_correct", "num_incorrect"]].sum().reset_index()

        df = df.groupby("title").apply(assess_history)
        df = df.sort_values("gs_max_time", ascending=True)

        del df["title"], df["gs_max_time"]

        if self.datatype == "test":
            df = pd.DataFrame([df.iloc[-1, :]])

        return df


def game_duration(val):
    val = json.loads(val)
    duration = val["duration"]
    g_misses = val["misses"]

    return [duration, g_misses]


def world_cum_duration_calc(world_df):
    # duration / missを抽出
    wg_df = world_df[(world_df.event_code == 2030) & (world_df.type == "Game")]
    du_miss = np.array(wg_df["event_data"].apply(game_duration).tolist())
    try:
        wg_df["duration"] = du_miss[:, 0]
        wg_df["misses"] = du_miss[:, 1]
    except BaseException:
        wg_df["duration"] = np.nan
        wg_df["misses"] = np.nan

    del du_miss

    aggs = {
        "duration": ["min", "mean", "max", "std", "count"],
        "misses": ["min", "mean", "max", "std"],
    }

    game_cums = groupings(
        wg_df, [
            "game_session", "gs_max_time", "world"], aggs, "g_")

    del wg_df
    gc.collect()

    # 累積を計算
    game_cums = game_cums.sort_values("gs_max_time").reset_index(drop=True)

    num_cols = [
        c for c in list(
            game_cums.columns) if c not in [
            "game_session",
            "gs_max_time",
            "world"]]
    cum_mean_cols = ["mean_" + c for c in num_cols]

    game_cums[cum_mean_cols] = game_cums[num_cols].cumsum()
    game_cums["cumnum"] = (game_cums.index + 1).values
    game_cums[cum_mean_cols] /= game_cums["cumnum"].values.reshape((-1, 1))

    game_cums[["game_session", "gs_max_time", "world"] + cum_mean_cols]

    # 直前のgameまでの累積結果をmergeする
    game_ass_uni = world_df[["world", "game_session", "type", "installation_id",
                             "gs_max_time"]].drop_duplicates().sort_values("gs_max_time").reset_index(drop=True)

    game_ass_uni = pd.merge(
        game_ass_uni, game_cums, how="left", on=[
            "game_session", "gs_max_time", "world"]).fillna(
        method="ffill")
    game_ass_uni = game_ass_uni.loc[game_ass_uni.type == "Assessment"]

    return game_ass_uni


class GameDurMiss(Features):
    """assessment 直前までのgameのプレイ状況を取得する
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

        use_cols = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy", "title",
                                                              "type", "event_code", "gs_max_time"
                                                              ]]
        self.format_and_save_feats(ret[use_cols])

        return ret[use_cols]

    def ins_id_sessions(self, df):
        """assessment 直前までのgameのプレイ状況を取得する
        Args:
            df: df grouped by installation_id
        """
        # 単純なactivity count
        gs_game_ass = df.loc[((df.event_code == 2030) & (
            df.type == "Game")) | (df.type == "Assessment")]
        gs_game_ass["gs_max_time"] = gs_game_ass.groupby(
            "game_session")["timestamp"].transform("max")

        game_ass_uni = gs_game_ass.groupby("world").apply(
            world_cum_duration_calc).reset_index(drop=True).sort_values("gs_max_time")

        del gs_game_ass

        if self.datatype == "test":
            game_ass_uni = pd.DataFrame([game_ass_uni.iloc[-1, :]])

        return game_ass_uni
