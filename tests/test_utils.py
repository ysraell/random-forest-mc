import pytest
import json
from pathlib import Path
import numpy as np
import datetime
from src.random_forest_mc.utils import (
    flat,
    flatten_nested_list,
    json_encoder,
    load_file_json,
    dump_file_json,
    LoadDicts,
    DEFAULT_DICT_PATH,
    JSON_EXTENSION,
)

# Test for flat function
def test_flat():
    assert flat([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flat([[1], [2, 3], []]) == [1, 2, 3]
    assert flat([]) == []
    assert flat([[]]) == []

# Test for flatten_nested_list function
def test_flatten_nested_list():
    assert flatten_nested_list([1, [2, 3], 4]) == [1, 2, 3, 4]
    assert flatten_nested_list([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert flatten_nested_list([]) == []
    assert flatten_nested_list([1, 2, 3]) == [1, 2, 3]
    assert flatten_nested_list([[[1]]]) == [1]

# Test for json_encoder function
def test_json_encoder_numpy_generic():
    assert json_encoder(np.int64(5)) == 5
    assert json_encoder(np.float32(3.14)) == 3.14

def test_json_encoder_datetime():
    dt_obj = datetime.datetime(2023, 1, 1, 10, 30, 0)
    assert json_encoder(dt_obj) == "2023-01-01T10:30:00"
    date_obj = datetime.date(2023, 1, 1)
    assert json_encoder(date_obj) == "2023-01-01"

def test_json_encoder_other_types():
    # Should return the object itself for types it doesn't handle
    assert json_encoder("test") == "test"
    assert json_encoder(123) == 123
    assert json_encoder({"a": 1}) == {"a": 1}

# Test for load_file_json and dump_file_json
def test_file_json_operations(tmp_path):
    test_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
    file_path = tmp_path / "test.json"

    dump_file_json(file_path, test_data)
    assert file_path.exists()

    loaded_data = load_file_json(file_path)
    assert loaded_data == test_data

def test_load_file_json_not_found(tmp_path):
    non_existent_file = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        load_file_json(non_existent_file)

def test_dump_file_json_with_numpy_and_datetime(tmp_path):
    data_with_special_types = {
        "np_int": np.int64(10),
        "dt_obj": datetime.datetime(2024, 7, 12),
    }
    file_path = tmp_path / "special_types.json"
    dump_file_json(file_path, data_with_special_types)
    loaded_data = load_file_json(file_path)
    assert loaded_data["np_int"] == 10
    assert loaded_data["dt_obj"] == "2024-07-12T00:00:00"

# Test for LoadDicts class
class TestLoadDicts:
    def setup_method(self, method):
        # Create a temporary directory for test JSON files
        self.test_dir = Path("test_dicts_data")
        self.test_dir.mkdir(exist_ok=True)

        self.file1_path = self.test_dir / "data1.json"
        self.file2_path = self.test_dir / "data2.json"
        self.invalid_file_path = self.test_dir / "invalid.json"
        self.keyword_file_path = self.test_dir / "class.json" # 'class' is a Python keyword

        with open(self.file1_path, "w") as f:
            json.dump({"key1": "value1"}, f)
        with open(self.file2_path, "w") as f:
            json.dump({"key2": "value2"}, f)
        with open(self.invalid_file_path, "w") as f:
            f.write("{invalid json}")
        with open(self.keyword_file_path, "w") as f:
            json.dump({"keyword_key": "keyword_value"}, f)

    def teardown_method(self, method):
        # Clean up the temporary directory
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_dicts_init_success(self):
        loader = LoadDicts(dict_path=self.test_dir)
        assert len(loader) == 3 # data1, data2, class
        assert "data1" in loader.List
        assert "data2" in loader.List
        assert "class" in loader.List
        assert loader.Dict["data1"] == {"key1": "value1"}
        assert loader.Dict["data2"] == {"key2": "value2"}
        assert loader.Dict["class"] == {"keyword_key": "keyword_value"}
        assert hasattr(loader, "data1")
        assert hasattr(loader, "data2")
        assert not hasattr(loader, "class") # 'class' is a keyword, so it shouldn't be an attribute
        assert "class" in loader.not_attr

    def test_load_dicts_init_ignore_errors(self):
        loader = LoadDicts(dict_path=self.test_dir, ignore_errors=True)
        # Should load valid files and ignore invalid one
        assert len(loader) == 3 # data1, data2, class
        assert "data1" in loader.List
        assert "data2" in loader.List
        assert "class" in loader.List
        assert loader.Dict["data1"] == {"key1": "value1"}
        assert loader.Dict["data2"] == {"key2": "value2"}
        assert loader.Dict["class"] == {"keyword_key": "keyword_value"}
        # The invalid file will cause an error during loading, but it should be ignored.
        # The `print` statement in the original code makes it hard to assert directly,
        # but we can check that the valid files are loaded.

    def test_load_dicts_init_raise_error(self):
        # If ignore_errors is False (default), it should raise an exception for invalid JSON
        # We need to create a new instance with only the invalid file to isolate the test
        temp_dir_for_error = Path("test_dicts_data_error")
        temp_dir_for_error.mkdir(exist_ok=True)
        invalid_file_path_isolated = temp_dir_for_error / "invalid_isolated.json"
        with open(invalid_file_path_isolated, "w") as f:
            f.write("{invalid json}")

        with pytest.raises(json.JSONDecodeError):
            LoadDicts(dict_path=temp_dir_for_error, ignore_errors=False)
        
        import shutil
        shutil.rmtree(temp_dir_for_error)


    def test_load_dicts_repr(self):
        loader = LoadDicts(dict_path=self.test_dir)
        # The order of files might vary, so check for substrings
        repr_str = str(loader)
        assert "LoadDicts:" in repr_str
        assert "data1" in repr_str
        assert "data2" in repr_str
        assert "class" in repr_str

    def test_load_dicts_len(self):
        loader = LoadDicts(dict_path=self.test_dir)
        assert len(loader) == 3

    def test_load_dicts_iter(self):
        loader = LoadDicts(dict_path=self.test_dir)
        items = list(loader)
        assert {"key1": "value1"} in items
        assert {"key2": "value2"} in items
        assert {"keyword_key": "keyword_value"} in items
        assert len(items) == 3

    def test_load_dicts_getitem(self):
        loader = LoadDicts(dict_path=self.test_dir)
        assert loader["data1"] == {"key1": "value1"}
        assert loader["data2"] == {"key2": "value2"}
        assert loader["class"] == {"keyword_key": "keyword_value"}
        with pytest.raises(KeyError):
            _ = loader["non_existent"]

    def test_load_dicts_items(self):
        loader = LoadDicts(dict_path=self.test_dir)
        items = dict(loader.items())
        assert items["data1"] == {"key1": "value1"}
        assert items["data2"] == {"key2": "value2"}
        assert items["class"] == {"keyword_key": "keyword_value"}
        assert len(items) == 3

    def test_load_dicts_add(self):
        # Create a first loader
        loader1 = LoadDicts(dict_path=self.test_dir)

        # Create a second temporary directory and loader
        test_dir2 = Path("test_dicts_data_2")
        test_dir2.mkdir(exist_ok=True)
        file3_path = test_dir2 / "data3.json"
        with open(file3_path, "w") as f:
            json.dump({"key3": "value3"}, f)
        loader2 = LoadDicts(dict_path=test_dir2)

        loader1.add(loader2)

        assert len(loader1) == 4
        assert "data1" in loader1.List
        assert "data2" in loader1.List
        assert "class" in loader1.List
        assert "data3" in loader1.List
        assert loader1.Dict["data3"] == {"key3": "value3"}

        import shutil
        shutil.rmtree(test_dir2)

    def test_default_dict_path_and_json_extension(self):
        assert DEFAULT_DICT_PATH == "./data"
        assert JSON_EXTENSION == ".json"
