# base code from https://github.com/iarai/NeurIPS2021-traffic4cast

import logging
import os


def t4c_apply_basic_logging_config(loglevel: str = None):
    logging.basicConfig(
        level=logging.INFO,#os.environ.get("LOGLEVEL", "INFO") if loglevel is None else loglevel,
        format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        force=True
    )

"""
https://stackoverflow.com/questions/32681289/python-logging-wont-set-a-logging-level-using-basicconfig
"""
