import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class worldLabelEncodingDiffFeatures(Features):
    """Event count in only Assessments
    """
    media_sequence = StringIO("""
title	type	media_duration
Welcome to Lost Lagoon!	Clip	19
Tree Top City - Level 1	Clip	17
Ordering Spheres	Clip	61
All Star Sorting	Game	
Costume Box	Clip	61
Fireworks (Activity)	Activity	
12 Monkeys	Clip	109
Tree Top City - Level 2	Clip	25
Flower Waterer (Activity)	Activity	
Pirate's Tale	Clip	80
Mushroom Sorter (Assessment)	Assessment	
Air Show	Game	
Treasure Map	Clip	156
Tree Top City - Level 3	Clip	26
Crystals Rule	Game	
Rulers	Clip	126
Bug Measurer (Activity)	Activity	
Bird Measurer (Assessment)	Assessment	
Magma Peak - Level 1	Clip	20
Sandcastle Builder (Activity)	Activity	
Slop Problem	Clip	60
Scrub-A-Dub	Game	
Watering Hole (Activity)	Activity	
Magma Peak - Level 2	Clip	22
Dino Drink	Game	
Bubble Bath	Game	
Bottle Filler (Activity)	Activity	
Dino Dive	Game	
Cauldron Filler (Assessment)	Assessment	
Crystal Caves - Level 1	Clip	18
Chow Time	Game	
Balancing Act	Clip	72
Chicken Balancer (Activity)	Activity	
Lifting Heavy Things	Clip	118
Crystal Caves - Level 2	Clip	24
Honey Cake	Clip	142
Happy Camel	Game	
Cart Balancer (Assessment)	Assessment	
Leaf Leader	Game	
Crystal Caves - Level 3	Clip	19
Heavy, Heavier, Heaviest	Clip	61
Pan Balance	Game	
Egg Dropper (Activity)	Activity	
Chest Sorter (Assessment)	Assessment	
""".strip())
    media_sequence = pd.read_csv(media_sequence, sep="\t")

    def __init__(self, train_labels, params, logger=None):
        super().__init__(params, logger=logger)
        self.train_labels = train_labels        

    def get_encoder(self, org_train, org_test):
        self.all_event_codes = set(org_train["event_code"].unique()).union(
            org_test["event_code"].unique())
        self.inverse_activities_map = self.media_sequence.title.to_dict()
        self.activities_map = {v: k for k, v in self.inverse_activities_map.items()}

    def calc_feature(self, org_train, org_test):
        if self.datatype == "train":
            df = org_train
            df = df.loc[df.installation_id.isin(
                self.train_labels.installation_id.unique())]
        else:
            # 直前までのnum_correct/incorrectを取得する
            df = org_test

        # get encodings informations
        self.get_encoder(org_train, org_test)

        ret = applyParallel(
                df.groupby("installation_id"),
                self._get_worldwise_label_diff_features)

        self.format_and_save_feats(ret)
        return ret

    def _get_worldwise_label_diff_features(self, df):
        """
        """
        df["title_enc"] = df["title"].map(self.activities_map)
        df = df.sort_values(['game_session', 'timestamp'])
        df = df.drop_duplicates(['installation_id', 'game_session'])

        res_df = df[['installation_id', 'game_session']].copy()

        for world in ['MAGMAPEAK', 'CRYSTALCAVES', 'TREETOPCITY', 'NONE']:
            _df = df.copy()
            _df.loc[_df.world != world, 'title_enc'] = None
            res_df[f'{world}_enc_diff_max'] = \
                    _df['title_enc'].rolling(window=len(_df), min_periods=1).apply(lambda x: pd.Series(x).dropna().diff().max())
            res_df[f'{world}_enc_diff_min'] = \
                    _df['title_enc'].rolling(window=len(_df), min_periods=1).apply(lambda x: pd.Series(x).dropna().diff().min())
            res_df[f'{world}_enc_diff_mean'] = \
                    _df['title_enc'].rolling(window=len(_df), min_periods=1).apply(lambda x: pd.Series(x).dropna().diff().mean())
            res_df[f'{world}_enc_diff_std'] = \
                    _df['title_enc'].rolling(window=len(_df), min_periods=1).apply(lambda x: pd.Series(x).dropna().diff().std())
            res_df[f'{world}_enc_just_before'] = df['title_enc'].apply(lambda x: pd.Series(x).dropna().iloc[-1])
            res_df[f'{world}_enc_diff_just_before'] = df['title_enc'] - res_df[f'{world}_enc_just_before']

        if self.datatype == "test":
            res_df = pd.DataFrame([res_df.iloc[-1, :]])
        else:
            # to save memory
            res_df = res_df[
                res_df.game_session
                .isin(self.train_labels.game_session)
            ]

        return res_df
