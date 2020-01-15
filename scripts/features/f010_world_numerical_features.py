import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class worldNumeriacalFeatures(Features):
    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def get_encoder(self, org_train, org_test):
        self.all_event_codes = set(org_train["event_code"].unique()).union(
            org_test["event_code"].unique())
        self.inverse_activities_map = self.media_sequence.title.to_dict()
        self.activities_map = {v: k for k,
                               v in self.inverse_activities_map.items()}

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
            self._calc_features)

        self.format_and_save_feats(ret)
        return ret

    def _calc_features(self, df):
        grp_df = df.groupby(['installation_id', 'game_session']).agg(
            {
                'num_correct': ['sum'],
                'num_incorrect': ['sum'],
                'game_time': ['max', 'std'],
                'event_count': ['max'],
#                 'title': {
#                     'Clip_num': lambda x: (x == "Clip").sum(),
#                     'Activity_num': lambda x: (x == "Activity").sum(),
#                     'Game_num': lambda x: (x == "Game").sum(),
#                     'Assessment_num': lambda x: (x == "Assessment").sum(),
#                     }
            }
        )

        grp_df.columns = [
            f'{col[0]}_{col[1]}' for col in grp_df.columns]

        grp_df['accuracy'] = grp_df['num_correct_sum'] / \
            (grp_df['num_correct_sum'] + grp_df['num_incorrect_sum'])

        res_df = grp_df[['installation_id', 'game_session']]

        for world in ['MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY', 'NONE']:
            _df = grp_df.copy()
            _df.loc[_df.world != world, 'title_enc'] = None

            res_df[f'{world}_accuracy_max'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).max()
            res_df[f'{world}_accuracy_min'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).min()
            res_df[f'{world}_accuracy_mean'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).mean()
            res_df[f'{world}_accuracy_std'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).std()
            res_df[f'{world}_just_before_accuracy'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            res_df[f'{world}_game_time_max_max'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).max()
            res_df[f'{world}_game_time_max_min'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).min()
            res_df[f'{world}_game_time_max_mean'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).mean()
            res_df[f'{world}_game_time_max_std'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).std()
            res_df[f'{world}_just_before_game_time_max'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            res_df[f'{world}_game_time_std_max'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).max()
            res_df[f'{world}_game_time_std_min'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).min()
            res_df[f'{world}_game_time_std_mean'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).mean()
            res_df[f'{world}_game_time_std_std'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).std()
            res_df[f'{world}_just_before_game_time_std'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            res_df[f'{world}_event_count_max_max'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).max()
            res_df[f'{world}_event_count_max_min'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).min()
            res_df[f'{world}_event_count_max_mean'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).mean()
            res_df[f'{world}_event_count_max_std'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).std()
            res_df[f'{world}_just_before_event_count_max'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

        if self.datatype == "test":
            res_df = pd.DataFrame([res_df.iloc[-1, :]])

        return res_df
