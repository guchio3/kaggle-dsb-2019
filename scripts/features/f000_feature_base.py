import gc
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

from yamakawa_san_utils import pickle_dump, timer


class Features(metaclass=ABCMeta):

    def __init__(self, params, logger=None):
        self.name = self.__class__.__name__
        self.datatype = params["datatype"]
        self.debug = params["debug"]
        self.is_overwrite = params["is_overwrite"]
        self.org_columns = []
        self.logger = logger

        self.input_dir = os.path.join(os.path.dirname("__file__"), "../input")
        self.df_path = Path(self.input_dir) / f"{self.datatype}.csv"

        #self.save_dir = os.path.join(
        #    os.path.dirname("__file__"),
        #    f"../../inputs/feature")
        self.save_dir = './mnt/inputs/features'
        self.save_type_dir = Path(self.save_dir) / f"{self.datatype}"
        self.save_path = Path(self.save_type_dir) / f"{self.name}.pkl"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_type_dir, exist_ok=True)

    def feature_extract(self, org_train, org_test):
        if self.check_feature_exec():
            with timer(f"FE: {self.name}", self.logger):
                a = self.calc_feature(org_train, org_test)
            return a

    @abstractmethod
    def calc_feature(self):
        """calc and save features
        Return: feature_df
        """
        raise NotImplementedError

    def format_and_save_feats(self, feat_df):
        """保存するカラムなど特徴量の形式を指定する
        """
        feat_cols = [
            c for c in list(
                feat_df.columns) if c not in self.org_columns]
        print(f'save feats to {self.save_path}')
        pickle_dump(feat_df[feat_cols], self.save_path)

        del feat_df
        gc.collect()

    def check_feature_exec(self):
        """
        すでに対象の特徴が存在するかどうかをcheckする
        Returns: bool (Falseなら特徴作成しない)

        """
        path = self.save_path

        if self.is_overwrite:
            print(f"overwrite features : {self.name}")
            return True
        else:
            if os.path.exists(path) is False:
                print(f"creates new file : {self.name}")
                return True

        print(f"file exists : {self.name}")
        return False


if __name__ == '__main__':
    features = Features()
