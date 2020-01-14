import gc
import os
from io import StringIO

import pandas as pd

from features.f000_feature_base import Features
from yamakawa_san_utils import applyParallel

FEATURE_ID = os.path.basename(__file__).split('_')[0]


class encodingTitleOrder(Features):
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
            assess_user = df.loc[df.type == "Assessment"].installation_id.unique()
            df = df.loc[df.installation_id.isin(assess_user)]
        else:
            # 直前までのnum_correct/incorrectを取得する
            org_test.loc[(org_test.event_code.isin([4100, 4110])) & (org_test["event_data"].str.contains("true")), 'num_correct'] = 1
            org_test.loc[(org_test.event_code.isin([4100, 4110])) & (org_test["event_data"].str.contains("false")), 'num_incorrect'] = 1    
            df = org_test

        # get encodings informations
        self.get_encoder(org_train, org_test)

        ret = applyParallel(df.groupby("installation_id"), self.ins_id_sessions)
        use_cols = [c for c in list(ret.columns) if c not in ["accuracy","accuracy_group","cum_accuracy","title",
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
        df = df.loc[df.type=="Assessment"]
        df = df[["installation_id", "game_session", "title_enc"]].drop_duplicates().reset_index(drop=True)

        if self.datatype=="test":
            df = pd.DataFrame([df.iloc[-1, :]])
        return df
