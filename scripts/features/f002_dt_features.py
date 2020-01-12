import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel, groupings


class dtFeatures(Features):
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
            self.get_dt_features)
        # ret_col = [c for c in list(ret.columns) if c not in ["accuracy", "accuracy_group", "cum_accuracy",
        #                                                     "game_session", "installation_id", "title",
        #                                                     "type"]]
        # ret[ret_col] = ret[ret_col].fillna(0)
        # 
        # use_cols = [c for c in list(ret.columns) if c not in []]
        # self.format_and_save_feats(ret[use_cols])
        self.format_and_save_feats(ret)

        return ret

    def get_dt_features(self, df):
        """session当該session直前までのactivityを示す
        Args:
            df: df grouped by installation_id
        """
        ass_idx = (((df.event_code == 4100)
                    & (df.type != "Assessment")
                    & (df.title != "Bird Measurer (Assessment)"))
                   | ((df.event_code == 4110)
                      & (df.type != "Assessment")
                      & (df.title == "Bird Measurer (Assessment)"))).values
        # 計算量削減のため、対象行に限定
        df = df[ass_idx]

        ass_dt = pd.to_datetime(df.timestamp)
        df['assesment_day'] = ass_dt.dt.day
        df['assesment_dayofweek'] = ass_dt.dt.dayofweek
        df['assesment_hour'] = ass_dt.dt.hour
        df['assesment_month'] = ass_dt.dt.month
        df['assesment_minute'] = ass_dt.dt.minute
#        df['assesment_'] = ass_dt.dt.

        df = df.set_index(['game_session', 'installation_id'])\
                .add_prefix('dt_')\
                .reset_index()
        return df
