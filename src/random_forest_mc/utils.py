import json
import numpy as np
from typing import Any
from keyword import iskeyword
from pathlib import Path

# For backward compatibility with 3.7
# from typing import TypeAlias


def np_encoder(object):
    if isinstance(object, np.generic):
        # Coverage trick!
        _ = None
        return object.item()


# DictsPathType: TypeAlias = str
DictsPathType = str


def load_file_json(path: DictsPathType):
    with open(path, "r") as f:
        return json.load(f)


def dump_file_json(path: DictsPathType, var: Any):
    with open(path, "w") as f:
        return json.dump(var, f, indent=4, default=np_encoder)


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

    def __len__(self):
        return len(self.List)

    def items(self):
        for item in self.List:
            yield item, self.Dict[item]

    def __repr__(self) -> str:
        return "LoadDicts: {}".format(", ".join(self.List))


# EOF
