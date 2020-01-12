import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel, groupings


class dateFeatures(Features):
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

        ass_idx = ((df.event_code == 4100)
                   & (df.title != "Bird Measurer (Assessment)"))
                  | ((df.event_code == 4110)
                   & (df.title == "Bird Measurer (Assessment)")

        df.loc[c_ass_idx, 'num_correct']=1
        df.loc[inc_ass_idx, 'num_incorrect']=1

        df["num_correct"].fillna(0, inplace=True)
        df["num_incorrect"].fillna(0, inplace=True)

        df=df.loc[(df.type == "Assessment")]

        pv=pd.pivot_table(
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

        pv.columns=[c[0] + "_" + c[1] if c[1] != "" else c[0]
                      for c in list(pv.columns)]
        pv_num_cols=[c for c in list(pv.columns) if "correct" in c]

        cum_cols=["cum_" + c for c in pv_num_cols]  # 累積列

        pv_cum_corr_cols=[
            c for c in cum_cols if "cum_num_correct_" in c]  # correct 列のみ
        pv_cum_incorr_cols=[
            c for c in cum_cols if "cum_num_incorrect_" in c]  # incorrect 列
        pv_cum_acc_cols=[
            "cum_acc_" +
            re.sub(
                'num_incorrect_',
                '',
                c) for c in pv_cum_incorr_cols]

        pv.reset_index(drop=True, inplace=True)

        pv[pv_num_cols]=pv[pv_num_cols].shift(1).fillna(0)
        pv[cum_cols]=pv[pv_num_cols].cumsum()

        pv[pv_cum_acc_cols]=pd.DataFrame(
            pv[pv_cum_corr_cols].values / (pv[pv_cum_corr_cols].values + pv[pv_cum_incorr_cols].values))

        del pv["gs_max_time"]

        if self.datatype == "test":
            pv=pd.DataFrame([pv.iloc[-1, :]])

        return pv
