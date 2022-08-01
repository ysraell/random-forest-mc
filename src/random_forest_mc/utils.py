import json
from glob import glob
from typing import Any
from typing import NewType

import numpy as np



DictsPathType = NewType("DictsPath", str)


def load_file_json(path: DictsPathType):
    with open(path, "r") as f:
        return json.load(f)


def dump_file_json(path: DictsPathType, var: Any):
    with open(path, "w") as f:
        return json.dump(var, f, indent=4)


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

    def __repr__(self) -> str:
        return "LoadDicts: {}".format(", ".join(self.List))


# EOF
