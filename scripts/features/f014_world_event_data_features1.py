import gc
import json
import os

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class worldEventDataFeatures1(Features):
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
            self._calc_features)

        self.format_and_save_feats(ret)
        return ret

    def _calc_features(self, df):
        df['level'] = df['event_data'].apply(
            lambda x: self._extract_key_from_json(x, 'level'))
        df['miss'] = df['event_data'].apply(
            lambda x: self._extract_key_from_json(x, 'miss'))
        df['round'] = df['event_data'].apply(
            lambda x: self._extract_key_from_json(x, 'round'))

        grp_df = df.groupby(['installation_id', 'game_session', 'world']).agg(
            {
                'level': {
                    'mean': lambda x: None if np.isnan(x).all() else x.mean(),
                    'std': lambda x: None if np.isnan(x).all() else x.std(),
                    'max': lambda x: None if np.isnan(x).all() else x.max(),
                    'min': lambda x: None if np.isnan(x).all() else x.min(),
                },
                'round': {
                    'mean': lambda x: None if np.isnan(x).all() else x.mean(),
                    'std': lambda x: None if np.isnan(x).all() else x.std(),
                    'max': lambda x: None if np.isnan(x).all() else x.max(),
                    'min': lambda x: None if np.isnan(x).all() else x.min(),
                },
                'misses': {
                    'mean': lambda x: None if np.isnan(x).all() else x.mean(),
                    'std': lambda x: None if np.isnan(x).all() else x.std(),
                    'max': lambda x: None if np.isnan(x).all() else x.max(),
                    'min': lambda x: None if np.isnan(x).all() else x.min(),
                },
            }
        )
        grp_df.columns = [
            f'{col[0]}_{col[1]}' for col in grp_df.columns]

        grp_df = grp_df.sort_values('timestamp_max')
        grp_df = grp_df.drop('timestamp_max', axis=1)
        grp_df = grp_df.shift(1)
        grp_df = grp_df.reset_index()

        res_df = grp_df[['installation_id', 'game_session']]

        worlds = ['MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY', 'NONE']
        for world in worlds:
            _df = grp_df.copy()
            for key in ['level', 'round', 'misses']:
                for stat in ['mean', 'std', 'max', 'min']:
                    key_stat = f'{key}_{stat}'
                    _df.loc[_df.world != world, key_stat] = None

                    res_df[f'{world}_{key_stat}_max'] = \
                        _df[key_stat].rolling(
                        window=len(_df), min_periods=1).max()
                    res_df[f'{world}_{key_stat}_min'] = \
                        _df[key_stat].rolling(
                        window=len(_df), min_periods=1).min()
                    res_df[f'{world}_{key_stat}_mean'] = \
                        _df[key_stat].rolling(
                        window=len(_df), min_periods=1).mean()
                    res_df[f'{world}_{key_stat}_std'] = \
                        _df[key_stat].rolling(
                        window=len(_df), min_periods=1).std()
                    res_df[f'{world}_just_before_{key_stat}'] = \
                        _df[key_stat].rolling(
                        window=len(_df), min_periods=1).apply(
                        lambda x: pd.Series(x).dropna().iloc[-1])

        for key in ['level', 'round', 'misses']:
            for stat in ['mean', 'std', 'max', 'min']:
                key_stat = f'{key}_{stat}'

                res_df[f'world_{key_stat}_mean'] = res_df[[
                    f'{world}_{key_stat}_mean' for world in worlds]].mean(axis=1).values
                res_df[f'world_{key_stat}_std'] = res_df[[
                    f'{world}_{key_stat}_std' for world in worlds]].std(axis=1).values
                res_df[f'world_{key_stat}_max'] = res_df[[
                    f'{world}_{key_stat}_max' for world in worlds]].max(axis=1).values
                res_df[f'world_{key_stat}_min'] = res_df[[
                    f'{world}_{key_stat}_max' for world in worlds]].max(axis=1).values

        if self.datatype == "test":
            res_df = pd.DataFrame([res_df.iloc[-1, :]])
        else:
            # to save memory
            res_df = res_df[
                res_df.game_session
                .isin(self.train_labels.game_session)
            ]

        return res_df

    def _extract_key_from_json(self, json_str, key):
        json_dict = json.loads(json_str)
        return json_dict[key] if key in json_dict else None
