import json
from glob import glob
from typing import NewType


DictsPathType = NewType("DictsPath", str)


def open_file_json(path):
    with open(path, "r") as f:
        return json.load(f)


class LoadDicts:
    def __init__(self, dict_path: DictsPathType = "./data"):
        Dicts_glob = glob(f"{dict_path}/*.json")
        self.List = []
        self.Dict = {}
        for path_json in Dicts_glob:
            name = path_json.split("/")[-1].replace(".json", "")
            self.List.append(name)
            self.Dict[name] = open_file_json(path_json)
            setattr(self, name, self.Dict[name])


# EOF
