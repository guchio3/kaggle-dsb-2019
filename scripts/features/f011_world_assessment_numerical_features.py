import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class worldAssessmentNumeriacalFeatures(Features):
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

        c_ass_idx = ((df.type == "Assessment")
                     & (df["event_data"].str.contains("true"))) | \
            ((df.type == "Assessment")
             & (df["event_data"].str.contains("true")))

        inc_ass_idx = ((df.type == "Assessment")
                       & (df["event_data"].str.contains("false"))) | \
            ((df.type == "Assessment")
             & (df["event_data"].str.contains("false")))

        df.loc[c_ass_idx, 'num_correct'] = 1
        df.loc[inc_ass_idx, 'num_incorrect'] = 1

        ret = applyParallel(
            df.groupby("installation_id"),
            self._calc_features)

        self.format_and_save_feats(ret)
        return ret

    def _calc_features(self, df):
        grp_df = df.groupby(['installation_id', 'game_session', 'world']).agg(
            {
                'timestamp': ['max', ],
                'num_correct': ['sum'],
                'num_incorrect': ['sum'],
                'game_time': ['max', 'std'],
                'event_count': ['max'],
            }
        )

        grp_df.columns = [
            f'{col[0]}_{col[1]}' for col in grp_df.columns]

        grp_df['accuracy'] = grp_df['num_correct_sum'] / \
            (grp_df['num_correct_sum'] + grp_df['num_incorrect_sum'])

        grp_df = grp_df.sort_values('timestamp_max')
        grp_df = grp_df.drop('timestamp_max', axis=1)
        grp_df = grp_df.shift(1)
        grp_df = grp_df.reset_index()

        res_df = grp_df[['installation_id', 'game_session']]

        for world in ['MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY', 'NONE']:
            _df = grp_df.copy()
            _df.loc[_df.world != world, 'accuracy'] = None

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

        worlds = ['MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY', 'NONE']
        res_df['world_accracy_mean_mean'] = res_df[[
            f'{world}_accuracy_mean' for world in worlds]].mean(axis=1).values
        res_df['world_accracy_mean_std'] = res_df[[
            f'{world}_accuracy_mean' for world in worlds]].std(axis=1).values
        res_df['world_accracy_max_max'] = res_df[[
            f'{world}_accuracy_max' for world in worlds]].max(axis=1).values
        res_df['world_accracy_max_mean'] = res_df[[
            f'{world}_accuracy_max' for world in worlds]].mean(axis=1).values
        res_df['world_accracy_max_std'] = res_df[[
            f'{world}_accuracy_max' for world in worlds]].std(axis=1).values
        res_df['world_accracy_min_min'] = res_df[[
            f'{world}_accuracy_max' for world in worlds]].min(axis=1).values
        res_df['world_accracy_min_mean'] = res_df[[
            f'{world}_accuracy_max' for world in worlds]].mean(axis=1).values
        res_df['world_accracy_min_std'] = res_df[[
            f'{world}_accuracy_max' for world in worlds]].std(axis=1).values
        res_df['world_accracy_std_mean'] = res_df[[
            f'{world}_accuracy_std' for world in worlds]].mean(axis=1).values
        res_df['world_accracy_std_std'] = res_df[[
            f'{world}_accuracy_std' for world in worlds]].std(axis=1).values

#         res_df = res_df\
#             .set_index(['installation_id', 'game_session'])\
#             .add_prefix(f'{FEATURE_ID}_worldwise_assessment_')\
#             .reset_index()

        if self.datatype == "test":
            res_df = pd.DataFrame([res_df.iloc[-1, :]])
        else:
            # to save memory
            res_df = res_df[
                res_df.game_session
                .isin(self.train_labels.game_session)
            ]

        return res_df
