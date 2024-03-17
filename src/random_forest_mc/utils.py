import json
from typing import Any, NewType
import numpy as np
from keyword import iskeyword
import datetime
from pathlib import Path


def json_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    if isinstance(object, (datetime.date, datetime.datetime)):
        return object.isoformat()


DictsPathType = NewType("DictsPath", str)


def load_file_json(path: DictsPathType):
    with open(Path(path), "r") as f:
        return json.load(f)


def dump_file_json(path: DictsPathType, var: Any):
    with open(Path(path), "w") as f:
        return json.dump(var, f, indent=4, default=json_encoder)


class LoadDicts:
    def __init__(
        self, dict_path: DictsPathType = "./data", ignore_errors: bool = False
    ):
        Dicts_glob = Path().glob(f"{dict_path}/*.json")
        self.List = []
        self.Dict = {}
        self.not_attr = []
        for path_json in Dicts_glob:
            try:
                name = path_json.as_posix().split("/")[-1].replace(".json", "")
                self.List.append(name)
                self.Dict[name] = load_file_json(path_json)
                if name.isidentifier() and not iskeyword(name):
                    setattr(self, name, self.Dict[name])
                else:
                    self.not_attr.append(name)
            except Exception as e:
                print(f"Error trying to load the file: {path_json.absolute()}: ")
                if not ignore_errors:
                    raise e
                print(e)

    def __repr__(self) -> str:
        return "LoadDicts: {}".format(", ".join(self.List))

    def __len__(self) -> int:
        return len(self.List)

    def __iter__(self) -> int:
        for item in self.List:
            yield self.Dict[item]

    def __getitem__(self, key):
        return self.Dict[key]

    def items(self):
        for item in self.List:
            yield item, self.Dict[item]


# EOF
