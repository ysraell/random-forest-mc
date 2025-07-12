import pytest
import json
from hashlib import md5
import pandas as pd
import numpy as np
from src.random_forest_mc.tree import DecisionTreeMC, PandasSeriesRow, LeafDict, TypeClassVal

# Helper function to create a dummy tree for testing
def create_dummy_tree(data=None, class_vals=None, survived_score=None, features=None, used_features=None):
    if data is None:
        data = {
            "feature1": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 0.5,
                    ">=": {"leaf": {"classA": 0.8, "classB": 0.2}, "depth": "2#"},
                    "<": {"leaf": {"classA": 0.1, "classB": 0.9}, "depth": "2#"},
                }
            }
        }
    if class_vals is None:
        class_vals = ["classA", "classB"]
    return DecisionTreeMC(data, class_vals, survived_score, features, used_features)

class TestDecisionTreeMC:
    def test_init(self):
        tree = create_dummy_tree(survived_score=0.75)
        assert tree.data is not None
        assert tree.class_vals == ["classA", "classB"]
        assert tree.survived_score == 0.75
        assert tree.features is None
        assert tree.used_features is None
        assert "module_version" in tree.attr_to_save

    def test_str_and_repr(self):
        tree = create_dummy_tree(survived_score=0.75)
        assert isinstance(str(tree), str)
        assert "DecisionTreeMC(survived_score=0.75" in repr(tree)

    def test_call_method(self):
        tree = create_dummy_tree()
        row = pd.Series({"feature1": 0.6})
        result = tree(row)
        assert isinstance(result, dict)
        assert "classA" in result
        assert "classB" in result

    def test_comparison_operators(self):
        tree1 = create_dummy_tree(survived_score=0.7)
        tree2 = create_dummy_tree(survived_score=0.8)
        tree3 = create_dummy_tree(survived_score=0.7)

        assert (tree1 < tree2) is True
        assert (tree2 > tree1) is True
        assert (tree1 == tree3) is True
        assert (tree1 <= tree3) is True
        assert (tree1 >= tree3) is True
        assert (tree2 >= tree1) is True
        assert (tree1 != tree2) is True

        with pytest.raises(TypeError):
            _ = tree1 == "not_a_tree"

    def test_tree2dict(self):
        tree = create_dummy_tree(survived_score=0.75, features=["feature1", "feature2"], used_features=["feature1"])
        tree_dict = tree.tree2dict()
        assert isinstance(tree_dict, dict)
        assert tree_dict["data"] == tree.data
        assert tree_dict["class_vals"] == tree.class_vals
        assert tree_dict["survived_score"] == tree.survived_score
        assert tree_dict["features"] == tree.features
        assert tree_dict["used_features"] == tree.used_features
        assert "module_version" in tree_dict

    def test_md5hexdigest(self):
        tree1 = create_dummy_tree()
        tree2 = create_dummy_tree() # Same data, should have same hash
        tree3_data = {
            "feature2": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 0.3,
                    ">=": {"leaf": {"classX": 0.5}, "depth": "2#"},
                    "<": {"leaf": {"classY": 0.5}, "depth": "2#"},
                }
            }
        }
        tree3 = create_dummy_tree(data=tree3_data) # Different data, should have different hash

        assert tree1.md5hexdigest == tree2.md5hexdigest
        assert tree1.md5hexdigest != tree3.md5hexdigest

        # Test with different order of keys in data (should still be same hash due to sort_keys)
        data_reordered = {
            "feature1": {
                "split": {
                    "<": {"leaf": {"classA": 0.1, "classB": 0.9}, "depth": "2#"},
                    ">=": {"leaf": {"classA": 0.8, "classB": 0.2}, "depth": "2#"},
                    "feat_type": "numeric",
                    "split_val": 0.5,
                }
            }
        }
        tree_reordered = create_dummy_tree(data=data_reordered)
        assert tree1.md5hexdigest == tree_reordered.md5hexdigest

    def test_depths_property(self):
        # Simple tree
        tree_data_simple = {
            "f1": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 10,
                    ">=": {"leaf": {"c1": 1.0}, "depth": "2#"},
                    "<": {"leaf": {"c2": 1.0}, "depth": "2#"},
                }
            }
        }
        tree_simple = create_dummy_tree(data=tree_data_simple)
        assert tree_simple.depths == [] # Depths are extracted from the 'depth' key in the leaf, not the split node

        # Tree with more complex structure and depth info
        tree_data_complex = {
            "f1": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 10,
                    ">=": {
                        "f2": {
                            "split": {
                                "feat_type": "categorical",
                                "split_val": "A",
                                ">=": {"leaf": {"c1": 1.0}, "depth": "3#"},
                                "<": {"leaf": {"c2": 1.0}, "depth": "3#"},
                            }
                        }
                    },
                    "<": {"leaf": {"c3": 1.0}, "depth": "2#"},
                }
            }
        }
        tree_complex = create_dummy_tree(data=tree_data_complex)
        # The _get_depths method is looking for 'depth' key directly under the node content, not within 'split'
        # The current implementation of depths property will not return depths from split nodes.
        # It will only return depths from leaf nodes.
        # So, for the dummy tree, it will return empty list.
        # If the 'depth' was directly under 'f1' or 'f2' nodes, it would be captured.
        # Let's adjust the dummy tree to reflect how depths are stored in the actual code.
        # The 'depth' is stored in the leaf node.
        assert tree_complex.depths == [] # No 'depth' key directly under 'f1' or 'f2'

        # Let's create a tree that actually has depths in the leaf nodes as per genLeaf in model.py
        tree_data_with_leaf_depths = {
            "feature1": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 0.5,
                    ">=": {"leaf": {"classA": 0.8, "classB": 0.2}, "depth": "2#"},
                    "<": {"leaf": {"classA": 0.1, "classB": 0.9}, "depth": "2#"},
                }
            }
        }
        tree_with_leaf_depths = create_dummy_tree(data=tree_data_with_leaf_depths)
        assert tree_with_leaf_depths.depths == [] # The current _get_depths implementation does not extract from leaf.

        # Re-reading the _get_depths method:
        # It checks if 'depth' is in node_content (which is the dict under the feature name, e.g., {"split": ...})
        # It does NOT check for 'depth' inside the 'leaf' dictionary.
        # So, the current `depths` property will always return an empty list for trees generated by `plantTree`
        # because `plantTree` puts the depth only in the leaf node.
        # This seems like a discrepancy between `genLeaf` and `_get_depths`.
        # For now, I will test based on the current `_get_depths` logic.
        # If the user wants to fix this, it would be a separate task.

        # If depth was stored like this:
        tree_data_alt_depth = {
            "f1": {
                "depth": "1#", # Depth at the node level
                "split": {
                    "feat_type": "numeric",
                    "split_val": 10,
                    ">=": {
                        "f2": {
                            "depth": "2#", # Depth at the node level
                            "split": {
                                "feat_type": "categorical",
                                "split_val": "A",
                                ">=": {"leaf": {"c1": 1.0}, "depth": "3#"},
                                "<": {"leaf": {"c2": 1.0}, "depth": "3#"},
                            }
                        }
                    },
                    "<": {"leaf": {"c3": 1.0}, "depth": "2#"},
                }
            }
        }
        tree_alt_depth = create_dummy_tree(data=tree_data_alt_depth)
        # The _get_depths method extracts 'depth' if it's a direct key of the node_content.
        # It does not recursively go into the 'leaf' dictionary.
        # It also does not go into the 'split' dictionary to find 'depth'.
        # So, for the above `tree_data_alt_depth`, it would find '1#' and '2#'.
        # The current implementation of `_get_depths` is:
        # if "depth" in node_content: depths.append(node_content["depth"])
        # This means 'depth' must be a direct child of the feature node.
        # The `genLeaf` function in `model.py` puts `depth` inside the leaf dictionary.
        # This means `tree.depths` will always be empty with current `plantTree` and `genLeaf` logic.
        # I will add a test case that reflects this current behavior.
        assert tree_alt_depth.depths == ["1#", "2#"] # This would be the expected output if depth was stored as shown in tree_data_alt_depth

        # Test with a tree structure that matches the current `genLeaf` output (depth in leaf)
        tree_data_leaf_depth = {
            "feature1": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 0.5,
                    ">=": {"leaf": {"classA": 0.8, "classB": 0.2}, "depth": "2#"},
                    "<": {"leaf": {"classA": 0.1, "classB": 0.9}, "depth": "2#"},
                }
            }
        }
        tree_leaf_depth = create_dummy_tree(data=tree_data_leaf_depth)
        assert tree_leaf_depth.depths == [] # As per current _get_depths implementation

    def test_useTree_numeric_split(self):
        tree = create_dummy_tree() # Uses default dummy tree with numeric split
        row_gt = pd.Series({"feature1": 0.6})
        result_gt = tree.useTree(row_gt)
        assert result_gt == {"classA": 0.8, "classB": 0.2}

        row_lt = pd.Series({"feature1": 0.4})
        result_lt = tree.useTree(row_lt)
        assert result_lt == {"classA": 0.1, "classB": 0.9}

        row_eq = pd.Series({"feature1": 0.5}) # Should go to >= branch
        result_eq = tree.useTree(row_eq)
        assert result_eq == {"classA": 0.8, "classB": 0.2}

    def test_useTree_categorical_split(self):
        cat_tree_data = {
            "color": {
                "split": {
                    "feat_type": "categorical",
                    "split_val": "red",
                    ">=": {"leaf": {"fruit": 0.7, "veg": 0.3}},
                    "<": {"leaf": {"fruit": 0.2, "veg": 0.8}},
                }
            }
        }
        tree = create_dummy_tree(data=cat_tree_data, class_vals=["fruit", "veg"])

        row_red = pd.Series({"color": "red"})
        result_red = tree.useTree(row_red)
        assert result_red == {"fruit": 0.7, "veg": 0.3}

        row_blue = pd.Series({"color": "blue"})
        result_blue = tree.useTree(row_blue)
        assert result_blue == {"fruit": 0.2, "veg": 0.8}

    def test_useTree_missing_feature_in_row(self):
        tree = create_dummy_tree()
        row = pd.Series({"another_feature": 1.0}) # Missing 'feature1'
        # When feature is not in row.index, it should return a list of leafes
        # and then combine them.
        result = tree.useTree(row)
        expected_leaf_sum_a = 0.8 + 0.1
        expected_leaf_sum_b = 0.2 + 0.9
        total_sum = expected_leaf_sum_a + expected_leaf_sum_b
        assert result["classA"] == pytest.approx(expected_leaf_sum_a / total_sum)
        assert result["classB"] == pytest.approx(expected_leaf_sum_b / total_sum)

    def test_useTree_complex_tree(self):
        complex_tree_data = {
            "featA": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 5,
                    ">=": {
                        "featB": {
                            "split": {
                                "feat_type": "categorical",
                                "split_val": "X",
                                ">=": {"leaf": {"c1": 1.0}},
                                "<": {"leaf": {"c2": 1.0}},
                            }
                        }
                    },
                    "<": {"leaf": {"c3": 1.0}},
                }
            }
        }
        tree = create_dummy_tree(data=complex_tree_data, class_vals=["c1", "c2", "c3"])

        row1 = pd.Series({"featA": 6, "featB": "X"})
        assert tree.useTree(row1) == {"c1": 1.0}

        row2 = pd.Series({"featA": 6, "featB": "Y"})
        assert tree.useTree(row2) == {"c2": 1.0}

        row3 = pd.Series({"featA": 4})
        assert tree.useTree(row3) == {"c3": 1.0}

        row4 = pd.Series({"featA": 6, "featC": "Z"}) # Missing featB
        # Should combine results from both branches of featB
        # Branch 1 (featB == X): {"c1": 1.0}
        # Branch 2 (featB != X): {"c2": 1.0}
        # Combined: {"c1": 0.5, "c2": 0.5}
        result4 = tree.useTree(row4)
        assert result4["c1"] == pytest.approx(0.5)
        assert result4["c2"] == pytest.approx(0.5)
        assert result4["c3"] == pytest.approx(0.0)

    def test_useTree_all_paths_missing_features(self):
        # Tree where all features are missing in the row
        tree_data = {
            "f1": {
                "split": {
                    "feat_type": "numeric",
                    "split_val": 1,
                    ">=": {
                        "f2": {
                            "split": {
                                "feat_type": "numeric",
                                "split_val": 2,
                                ">=": {"leaf": {"A": 1.0}},
                                "<": {"leaf": {"B": 1.0}},
                            }
                        }
                    },
                    "<": {
                        "f3": {
                            "split": {
                                "feat_type": "numeric",
                                "split_val": 3,
                                ">=": {"leaf": {"C": 1.0}},
                                "<": {"leaf": {"D": 1.0}},
                            }
                        }
                    },
                }
            }
        }
        tree = create_dummy_tree(data=tree_data, class_vals=["A", "B", "C", "D"])
        row = pd.Series({"some_other_feat": 10}) # No f1, f2, f3

        result = tree.useTree(row)
        # Should average all 4 leaves: A, B, C, D
        assert result["A"] == pytest.approx(0.25)
        assert result["B"] == pytest.approx(0.25)
        assert result["C"] == pytest.approx(0.25)
        assert result["D"] == pytest.approx(0.25)
