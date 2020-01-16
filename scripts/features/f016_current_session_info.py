import gc
import os
from io import StringIO

import numpy as np
import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class currentSessionInfo(Features):
    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels

    def get_encoder(self, org_train, org_test):
        self.all_titles = np.sort(list(set(org_train["title"].unique()).union(
            org_test["title"].unique())))
        self.titles_map = dict(
            zip(self.all_titles, np.arange(len(self.all_titles))))

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            df = df.loc[df.installation_id.isin(
                self.train_labels.installation_id.unique())]
        else:
            # 直前までのnum_correct/incorrectを取得する
            df = org_test

        self.get_encoder(org_train, org_test)

        df = df.drop_duplicates(['installation_id', 'game_session'])
        df = df.sort_values('timestamp')

        if self.datatype == "test":
            df = pd.DataFrame([df.iloc[-1, :]])
        else:
            df = df[
                df
                .game_session
                .isin(self.train_labels.game_session)
            ]

        ret = df[['installation_id', 'game_session']]
        # ret['title_LE'] = df.title.map(self.titles_map)
        ret['world_LE'] = df.world.map({
                        'MAGMAPEAK': 0,
                        'TREETOPCITY': 1,
                        'CRYSTALCAVES': 2,
                        'NONE': 3,
                    })

        ret = ret\
            .set_index(['installation_id', 'game_session'])\
            .add_prefix(f'{FEATURE_ID}_current_info_')\
            .reset_index()

        self.format_and_save_feats(ret)
        return ret
