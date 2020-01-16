import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class sameWorldBaseFeatures(Features):
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

        target_c_ass_idx = ((df.event_code == 4100)
                            & (df.title != "Bird Measurer (Assessment)")
                            & (df.type == "Assesment")
                            & (df["event_data"].str.contains("true"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df.type == "Assesment")
             & (df["event_data"].str.contains("true")))

        target_inc_ass_idx = ((df.event_code == 4100)
                              & (df.title != "Bird Measurer (Assessment)")
                              & (df.type == "Assesment")
                              & (df["event_data"].str.contains("false"))) | \
            ((df.event_code == 4110)
             & (df.title == "Bird Measurer (Assessment)")
             & (df.type == "Assesment")
             & (df["event_data"].str.contains("false")))

        assessment_c_ass_idx = ((df.type == "Assesment")
                                & (df["event_data"].str.contains("true"))) | \
            ((df.type == "Assesment")
             & (df["event_data"].str.contains("true")))

        assessment_inc_ass_idx = ((df.type == "Assesment")
                                  & (df["event_data"].str.contains("false"))) | \
            ((df.type == "Assesment")
             & (df["event_data"].str.contains("false")))

        activity_c_ass_idx = ((df.type == "Activity")
                              & (df["event_data"].str.contains("true"))) | \
            ((df.type == "Activity")
             & (df["event_data"].str.contains("true")))

        activity_inc_ass_idx = ((df.type == "Activity")
                                & (df["event_data"].str.contains("false"))) | \
            ((df.type == "Activity")
             & (df["event_data"].str.contains("false")))

        game_c_ass_idx = ((df.type == "Game")
                          & (df["event_data"].str.contains("true"))) | \
            ((df.type == "Game")
             & (df["event_data"].str.contains("true")))

        game_inc_ass_idx = ((df.type == "Game")
                            & (df["event_data"].str.contains("false"))) | \
            ((df.type == "Game")
             & (df["event_data"].str.contains("false")))

        df.loc[target_c_ass_idx, 'num_correct'] = 1
        df.loc[target_inc_ass_idx, 'num_incorrect'] = 1
        df.loc[assessment_c_ass_idx, 'assessment_num_correct'] = 1
        df.loc[assessment_inc_ass_idx, 'assessment_num_incorrect'] = 1
        df.loc[activity_c_ass_idx, 'activity_num_correct'] = 1
        df.loc[activity_inc_ass_idx, 'activity_num_incorrect'] = 1
        df.loc[game_c_ass_idx, 'game_num_correct'] = 1
        df.loc[game_inc_ass_idx, 'game_num_incorrect'] = 1

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
                'assessment_num_correct': ['sum'],
                'assessment_num_incorrect': ['sum'],
                'activity_num_correct': ['sum'],
                'activity_num_incorrect': ['sum'],
                'game_num_correct': ['sum'],
                'game_num_incorrect': ['sum'],
                'game_time': ['max', 'std'],
                'event_count': ['max'],
            }
        )

        grp_df.columns = [
            f'{col[0]}_{col[1]}' for col in grp_df.columns]

        grp_df['accuracy'] = grp_df['num_correct_sum'] / \
            (grp_df['num_correct_sum'] + grp_df['num_incorrect_sum'])
        grp_df['assessment_accuracy'] = grp_df['assessment_num_correct_sum'] / \
            (grp_df['assessment_num_correct_sum'] + grp_df['assessment_num_incorrect_sum'])
        grp_df['activity_accuracy'] = grp_df['activity_num_correct_sum'] / \
            (grp_df['activity_num_correct_sum'] + grp_df['activity_num_incorrect_sum'])
        grp_df['game_accuracy'] = grp_df['game_num_correct_sum'] / \
            (grp_df['game_num_correct_sum'] + grp_df['game_num_incorrect_sum'])

        grp_df = grp_df.sort_values('timestamp_max')
        grp_df = grp_df.drop('timestamp_max', axis=1)
        grp_df = grp_df.shift(1)
        grp_df = grp_df.reset_index()

        temp_df = grp_df[['installation_id', 'game_session', 'world']]
        res_df = grp_df[['installation_id', 'game_session']]

        for world in ['MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY', 'NONE']:
            _df = grp_df.copy()
            _df.loc[_df.world != world, 'accuracy'] = None
            _df.loc[_df.world != world, 'assessment_accuracy'] = None
            _df.loc[_df.world != world, 'activity_accuracy'] = None
            _df.loc[_df.world != world, 'game_accuracy'] = None
            _df.loc[_df.world != world, 'game_time_max'] = None
            _df.loc[_df.world != world, 'game_time_std'] = None
            _df.loc[_df.world != world, 'event_count_max'] = None

            temp_df[f'{world}_accuracy_max'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_accuracy_min'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_accuracy_mean'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_accuracy_std'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_just_before_accuracy'] = \
                _df['accuracy'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            temp_df[f'{world}_assessment_accuracy_max'] = \
                _df['assessment_accuracy'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_assessment_accuracy_min'] = \
                _df['assessment_accuracy'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_assessment_accuracy_mean'] = \
                _df['assessment_accuracy'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_assessment_accuracy_std'] = \
                _df['assessment_accuracy'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_assessment_just_before_accuracy'] = \
                _df['assessment_accuracy'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            temp_df[f'{world}_activity_accuracy_max'] = \
                _df['activity_accuracy'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_activity_accuracy_min'] = \
                _df['activity_accuracy'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_activity_accuracy_mean'] = \
                _df['activity_accuracy'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_activity_accuracy_std'] = \
                _df['activity_accuracy'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_activity_just_before_accuracy'] = \
                _df['activity_accuracy'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            temp_df[f'{world}_game_accuracy_max'] = \
                _df['game_accuracy'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_game_accuracy_min'] = \
                _df['game_accuracy'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_game_accuracy_mean'] = \
                _df['game_accuracy'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_game_accuracy_std'] = \
                _df['game_accuracy'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_game_just_before_accuracy'] = \
                _df['game_accuracy'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            temp_df[f'{world}_game_time_max_max'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_game_time_max_min'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_game_time_max_mean'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_game_time_max_std'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_just_before_game_time_max'] = \
                _df['game_time_max'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            temp_df[f'{world}_game_time_std_max'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_game_time_std_min'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_game_time_std_mean'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_game_time_std_std'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_just_before_game_time_std'] = \
                _df['game_time_std'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

            temp_df[f'{world}_event_count_max_max'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).max()
            temp_df[f'{world}_event_count_max_min'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).min()
            temp_df[f'{world}_event_count_max_mean'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).mean()
            temp_df[f'{world}_event_count_max_std'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).std()
            temp_df[f'{world}_just_before_event_count_max'] = \
                _df['event_count_max'].rolling(
                window=len(_df), min_periods=1).apply(
                lambda x: pd.Series(x).dropna().iloc[-1])

        if self.datatype == "test":
            res_df = pd.DataFrame([res_df.iloc[-1, :]])
            temp_df = pd.DataFrame([temp_df.iloc[-1, :]])
        else:
            res_df = res_df[
                res_df.game_session
                .isin(self.train_labels.game_session)
            ]
            temp_df = temp_df[
                temp_df.game_session
                .isin(self.train_labels.game_session)
            ]

        for i, row in res_df.iterrows():
            temp_row = temp_df.loc[i]
            world = temp_row['world']

            res_df.loc[i, 'world_accuracy_max'] = temp_row[f'{world}_accuracy_max']
            res_df.loc[i, 'world_accuracy_min'] = temp_row[f'{world}_accuracy_min']
            res_df.loc[i, 'world_accuracy_mean'] = temp_row[f'{world}_accuracy_mean']
            res_df.loc[i, 'world_accuracy_std'] = temp_row[f'{world}_accuracy_std']
            res_df.loc[i, 'world_just_before_accuracy'] = temp_row[f'{world}_just_before_accuracy']

            res_df.loc[i, 'world_assessment_accuracy_max'] = temp_row[f'{world}_assessment_accuracy_max']
            res_df.loc[i, 'world_assessment_accuracy_min'] = temp_row[f'{world}_assessment_accuracy_min']
            res_df.loc[i, 'world_assessment_accuracy_mean'] = temp_row[f'{world}_assessment_accuracy_mean']
            res_df.loc[i, 'world_assessment_accuracy_std'] = temp_row[f'{world}_assessment_accuracy_std']
            res_df.loc[i, 'world_assessment_just_before_accuracy'] = temp_row[f'{world}_assessment_just_before_accuracy']

            res_df.loc[i, 'world_activity_accuracy_max'] = temp_row[f'{world}_activity_accuracy_max']
            res_df.loc[i, 'world_activity_accuracy_min'] = temp_row[f'{world}_activity_accuracy_min']
            res_df.loc[i, 'world_activity_accuracy_mean'] = temp_row[f'{world}_activity_accuracy_mean']
            res_df.loc[i, 'world_activity_accuracy_std'] = temp_row[f'{world}_activity_accuracy_std']
            res_df.loc[i, 'world_activity_just_before_accuracy'] = temp_row[f'{world}_activity_just_before_accuracy']

            res_df.loc[i, 'world_game_accuracy_max'] = temp_row[f'{world}_game_accuracy_max']
            res_df.loc[i, 'world_game_accuracy_min'] = temp_row[f'{world}_game_accuracy_min']
            res_df.loc[i, 'world_game_accuracy_mean'] = temp_row[f'{world}_game_accuracy_mean']
            res_df.loc[i, 'world_game_accuracy_std'] = temp_row[f'{world}_game_accuracy_std']
            res_df.loc[i, 'world_game_just_before_accuracy'] = temp_row[f'{world}_game_just_before_accuracy']

            res_df.loc[i, 'world_game_time_max_max'] = temp_row[f'{world}_game_time_max_max']
            res_df.loc[i, 'world_game_time_max_min'] = temp_row[f'{world}_game_time_max_min']
            res_df.loc[i, 'world_game_time_max_mean'] = temp_row[f'{world}_game_time_max_mean']
            res_df.loc[i, 'world_game_time_max_std'] = temp_row[f'{world}_game_time_max_std']
            res_df.loc[i, 'world_just_before_game_time_max'] = temp_row[f'{world}_just_before_game_time_max']

            res_df.loc[i, 'world_game_time_std_max'] = temp_row[f'{world}_game_time_std_max']
            res_df.loc[i, 'world_game_time_std_min'] = temp_row[f'{world}_game_time_std_min']
            res_df.loc[i, 'world_game_time_std_mean'] = temp_row[f'{world}_game_time_std_mean']
            res_df.loc[i, 'world_game_time_std_std'] = temp_row[f'{world}_game_time_std_std']
            res_df.loc[i, 'world_just_before_game_time_std'] = temp_row[f'{world}_just_before_game_time_std']

            res_df.loc[i, 'world_event_count_max_max'] = temp_row[f'{world}_event_count_max_max']
            res_df.loc[i, 'world_event_count_max_min'] = temp_row[f'{world}_event_count_max_min']
            res_df.loc[i, 'world_event_count_max_mean'] = temp_row[f'{world}_event_count_max_mean']
            res_df.loc[i, 'world_event_count_max_std'] = temp_row[f'{world}_event_count_max_std']
            res_df.loc[i, 'world_just_before_event_count_max'] = temp_row[f'{world}_just_before_event_count_max']

        res_df = res_df\
            .set_index(['installation_id', 'game_session'])\
            .add_prefix(f'{FEATURE_ID}_same_world_base_')\
            .reset_index()

        return res_df
