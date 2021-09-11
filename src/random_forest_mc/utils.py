import json
from glob import glob
from typing import Any
from typing import NewType

import numpy as np


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


DictsPathType = NewType("DictsPath", str)


def load_file_json(path: DictsPathType):
    with open(path, "r") as f:
        return json.load(f)


def dump_file_json(path: DictsPathType, var: Any):
    with open(path, "w") as f:
        return json.dump(var, f, indent=4, default=np_encoder)


class LoadDicts:
    def __init__(self, dict_path: DictsPathType = "./data"):
        Dicts_glob = glob(f"{dict_path}/*.json")
        self.List = []
        self.Dict = {}
        for path_json in Dicts_glob:
            name = path_json.split("/")[-1].replace(".json", "")
            self.List.append(name)
            self.Dict[name] = load_file_json(path_json)
            setattr(self, name, self.Dict[name])


# EOF
