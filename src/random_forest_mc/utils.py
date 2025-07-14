import json
from typing import Any, NewType, Iterator, Tuple
import numpy as np
from keyword import iskeyword
import datetime
from pathlib import Path
import functools
import operator


def flat(a: list[list[Any]]) -> list[Any]:
    return functools.reduce(operator.iconcat, a, [])


def flatten_nested_list(lst: list[Any]) -> list[Any]:
    """
    Flatten a multi-level nested list into a single level list.

    Args:
        lst: A list that may contain other lists and non-list elements.

    Returns:
        A new list with all elements from the original list, flattened into one level.
    """
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    return result


def json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()


DEFAULT_DICT_PATH = "./data"
JSON_EXTENSION = ".json"

DictsPathType = NewType("DictsPath", str)


def load_file_json(path: DictsPathType):
    with open(Path(path), "r") as f:
        return json.load(f)


def dump_file_json(path: DictsPathType, var: Any):
    with open(Path(path), "w") as f:
        return json.dump(var, f, indent=4, default=json_encoder)


class LoadDicts:
    def __init__(self, dict_path: DictsPathType = DEFAULT_DICT_PATH, ignore_errors: bool = False):
        Dicts_glob = Path(dict_path).glob("*.json")
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

    def __iter__(self) -> Iterator[Any]:
        for item in self.List:
            yield self.Dict[item]

    def __getitem__(self, key: str) -> Any:
        return self.Dict[key]

    def items(self) -> Iterator[Tuple[str, Any]]:
        for item in self.List:
            yield item, self.Dict[item]

    def add(self, other: "LoadDicts") -> None:
        for item_name in other.List:
            self.Dict[item_name] = other.Dict[item_name]
        self.List.extend(other.List)


# EOF
