import os
import importlib


# set configs
EXP_ID = os.path.basename(__file__).split('_')[0]

CONFIG_MODULE = f'configs.{EXP_ID}_config'
CONFIG_DICT = importlib.import_module(CONFIG_MODULE).CONFIG_DICT



