import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.random_forest_mc.model import (
    RandomForestMC,
    MissingValuesNotFound,
    DatasetNotFound,
    DictValuesAllFeaturesMissing,
)
from src.random_forest_mc.tree import DecisionTreeMC


# Helper function to create a dummy DecisionTreeMC for mocking
def create_mock_tree(
    survived_score=0.5, data=None, class_vals=None, features=None, used_features=None
):
    mock_tree = MagicMock(spec=DecisionTreeMC)
    mock_tree.survived_score = survived_score
    mock_tree.data = data if data is not None else {"leaf": {"classA": 1.0}}
    mock_tree.class_vals = class_vals if class_vals is not None else ["classA"]
    mock_tree.features = features if features is not None else ["feature1"]
    mock_tree.used_features = (
        used_features if used_features is not None else ["feature1"]
    )
    mock_tree.md5hexdigest = "dummy_hash"
    mock_tree.useTree.return_value = {
        "classA": 1.0,
        "classB": 0.0,
    }  # Default return for tree call
    return mock_tree


class TestRandomForestMC:
    def test_init(self):
        rf = RandomForestMC()
        assert rf.n_trees == 16
        assert rf.target_col == "target"
        assert rf.batch_train_pclass == 10
        assert rf.batch_val_pclass == 10
        assert rf.max_discard_trees == 10
        assert rf.delta_th == 0.1
        assert rf.th_start == 1.0
        assert rf.get_best_tree is True
        assert rf.min_feature is None
        assert rf.max_feature is None
        assert rf.temporal_features is False
        assert rf.split_with_replace is False
        assert (
            rf.max_depth is not None
        )  # Should be set to recursion limit or int(max_depth)
        assert rf.min_samples_split == 1
        assert rf.got_best_tree_verbose is False

    def test_process_dataset(self):
        rf = RandomForestMC()
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": ["A", "B", "A", "B"],
                "target": ["yes", "no", "yes", "no"],
            }
        )
        rf.process_dataset(df)

        assert rf.numeric_cols == ["feature1"]
        assert sorted(rf.feature_cols) == sorted(["feature1", "feature2"])
        assert rf.type_of_cols == {"feature1": "numeric", "feature2": "categorical"}
        assert rf.dataset.equals(df)  # No NaNs in this df, so it should be the same
        assert sorted(rf.class_vals) == sorted(["yes", "no"])
        assert rf.min_feature == 2
        assert rf.max_feature == 2

        # Test with NaNs
        df_nan = pd.DataFrame(
            {
                "feature1": [1, 2, np.nan, 4],
                "target": ["yes", "no", "yes", "no"],
            }
        )
        rf_nan = RandomForestMC()
        rf_nan.process_dataset(df_nan)
        assert rf_nan.dataset.shape[0] == 3  # One row dropped

        # Test temporal features warning
        df_temporal = pd.DataFrame(
            {
                "feat_1": [1, 2],
                "feat_abc": [3, 4],
                "target": ["yes", "no"],
            }
        )
        rf_temporal = RandomForestMC(temporal_features=True)
        with patch("src.random_forest_mc.model.log") as mock_log:
            rf_temporal.process_dataset(df_temporal)
            mock_log.warning.assert_called_with(
                "Temporal features ordering disable: you do not have all orderable features!"
            )
            assert rf_temporal.temporal_features is False

    def test_split_train_val(self):
        rf = RandomForestMC(batch_train_pclass=1, batch_val_pclass=1)
        df = pd.DataFrame(
            {
                "feature1": range(10),
                "target": ["yes", "no"] * 5,
            }
        )
        rf.process_dataset(df)

        ds_T, ds_V = rf.split_train_val()
        assert ds_T.shape[0] == 2  # 1 yes, 1 no
        assert ds_V.shape[0] == 2  # 1 yes, 1 no
        assert "target" in ds_T.columns
        assert "target" in ds_V.columns

        # Test dropping single-valued columns in ds_T
        df_single_val = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1],
                "feature2": [2, 3, 4, 5],
                "target": ["yes", "no", "yes", "no"],
            }
        )
        rf_single_val = RandomForestMC()
        rf_single_val.process_dataset(df_single_val)
        ds_T_single, _ = rf_single_val.split_train_val()
        assert "feature1" not in ds_T_single.columns  # Should be dropped
        assert "feature2" in ds_T_single.columns

    def test_sampleFeats(self):
        rf = RandomForestMC(min_feature=1, max_feature=2)
        rf.target_col = "target"
        feature_cols = ["f1", "f2", "f3", "target"]

        with patch("numpy.random.randint", return_value=2):
            with patch("numpy.random.sample", return_value=np.array(["f1", "f3"])):
                sampled_feats = rf.sampleFeats(feature_cols.copy())
                assert sorted(sampled_feats) == sorted(["f1", "f3"])

        rf_temporal = RandomForestMC(
            min_feature=1, max_feature=3, temporal_features=True
        )
        rf_temporal.target_col = "target"
        temporal_feature_cols = ["feat_1", "feat_3", "feat_2", "target"]
        with patch("numpy.random.randint", return_value=2):
            with patch(
                "numpy.random.sample", return_value=np.array(["feat_3", "feat_1"])
            ):
                sampled_feats_temporal = rf_temporal.sampleFeats(
                    temporal_feature_cols.copy()
                )
                assert sampled_feats_temporal == [
                    "feat_1",
                    "feat_3",
                ]  # Should be sorted

    def test_genLeaf(self):
        rf = RandomForestMC(target_col="label")
        df = pd.DataFrame({"label": ["A", "B", "A", "C"]})
        leaf = rf.genLeaf(df, 3)
        assert leaf["depth"] == "3#"
        assert leaf["leaf"] == {"A": 0.5, "B": 0.25, "C": 0.25}

    def test_splitData_numeric(self):
        rf = RandomForestMC(target_col="target")
        rf.numeric_cols = ["feat1"]
        df = pd.DataFrame(
            {"feat1": [1, 2, 3, 4, 5], "target": ["a", "b", "a", "b", "a"]}
        )

        ds_a, ds_b, split_val = rf.splitData("feat1", df)
        assert split_val == 3.0  # Median of [1,2,3,4,5]
        assert ds_a["feat1"].tolist() == [3, 4, 5]
        assert ds_b["feat1"].tolist() == [1, 2]

        # Test edge case where one split is empty
        df_edge = pd.DataFrame(
            {"feat1": [1, 1, 1, 1, 1], "target": ["a", "b", "a", "b", "a"]}
        )
        ds_a_edge, ds_b_edge, split_val_edge = rf.splitData("feat1", df_edge)
        assert ds_a_edge["feat1"].tolist() == []  # No value > 1
        assert ds_b_edge["feat1"].tolist() == [1, 1, 1, 1, 1]  # All values <= 1

    def test_splitData_categorical(self):
        rf = RandomForestMC(target_col="target")
        rf.numeric_cols = []
        df = pd.DataFrame(
            {"feat1": ["red", "blue", "red", "green"], "target": ["a", "b", "a", "b"]}
        )

        ds_a, ds_b, split_val = rf.splitData("feat1", df)
        assert split_val == "red"  # Most common category
        assert ds_a["feat1"].tolist() == ["red", "red"]
        assert ds_b["feat1"].tolist() == ["blue", "green"]

    def test_splitData_small_df(self):
        rf = RandomForestMC(target_col="target")
        rf.numeric_cols = ["feat1"]
        df = pd.DataFrame({"feat1": [1, 2], "target": ["a", "b"]})
        ds_a, ds_b, split_val = rf.splitData("feat1", df)
        assert ds_a["feat1"].tolist() == [2]
        assert ds_b["feat1"].tolist() == [1]
        assert split_val == 2

    @patch("src.random_forest_mc.model.DecisionTreeMC")
    def test_plantTree(self, MockDecisionTreeMC):
        rf = RandomForestMC(target_col="target", max_depth=2, min_samples_split=1)
        rf.type_of_cols = {"f1": "numeric"}
        rf.class_vals = ["A", "B"]
        df_train = pd.DataFrame({"f1": [1, 2, 3, 4], "target": ["A", "B", "A", "B"]})

        # Mock the internal growTree function to control its output
        with patch.object(
            rf, "genLeaf", return_value={"leaf": {"A": 1.0}, "depth": "2#"}
        ):
            with patch.object(
                rf,
                "splitData",
                return_value=(df_train.iloc[:2], df_train.iloc[2:], 2.5),
            ):
                tree = rf.plantTree(df_train, ["f1"])
                MockDecisionTreeMC.assert_called_once()
                assert isinstance(tree, MockDecisionTreeMC)

    def test_validationTree(self):
        rf = RandomForestMC(target_col="target")
        rf.class_vals = ["A", "B"]
        mock_tree = create_mock_tree()
        mock_tree.__call__.side_effect = [
            {"A": 0.9, "B": 0.1},
            {"A": 0.2, "B": 0.8},
        ]  # Predict A, then B

        df_val = pd.DataFrame({"feature1": [1, 2], "target": ["A", "B"]})
        accuracy = rf.validationTree(mock_tree, df_val)
        assert accuracy == 1.0  # Both predictions are correct

        mock_tree.__call__.side_effect = [
            {"A": 0.9, "B": 0.1},
            {"A": 0.9, "B": 0.1},
        ]  # Predict A, then A
        df_val_partial = pd.DataFrame({"feature1": [1, 2], "target": ["A", "B"]})
        accuracy_partial = rf.validationTree(mock_tree, df_val_partial)
        assert accuracy_partial == 0.5  # Only first prediction is correct

    @patch("src.random_forest_mc.model.RandomForestMC.split_train_val")
    @patch("src.random_forest_mc.model.RandomForestMC.sampleFeats")
    @patch("src.random_forest_mc.model.RandomForestMC.plantTree")
    @patch("src.random_forest_mc.model.RandomForestMC.validationTree")
    def test_survivedTree(
        self,
        mock_validationTree,
        mock_plantTree,
        mock_sampleFeats,
        mock_split_train_val,
    ):
        rf = RandomForestMC(
            th_start=0.8, delta_th=0.1, max_discard_trees=3, get_best_tree=True
        )
        rf.feature_cols = ["f1", "f2"]

        mock_split_train_val.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_sampleFeats.return_value = ["f1"]

        # Scenario 1: Tree immediately survives
        mock_plantTree.return_value = create_mock_tree(survived_score=0.9)
        mock_validationTree.return_value = 0.9
        tree = rf.survivedTree()
        assert tree.survived_score == 0.9
        assert mock_validationTree.call_count == 1

        # Scenario 2: Tree fails, then best tree is kept
        rf_best = RandomForestMC(
            th_start=0.8, delta_th=0.1, max_discard_trees=1, get_best_tree=True
        )
        rf_best.feature_cols = ["f1", "f2"]
        mock_split_train_val.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_sampleFeats.return_value = ["f1"]
        mock_plantTree.side_effect = [
            create_mock_tree(survived_score=0.6),  # First tree, fails
            create_mock_tree(survived_score=0.7),  # Second tree, fails but is better
        ]
        mock_validationTree.side_effect = [0.6, 0.7]

        tree_best = rf_best.survivedTree()
        assert tree_best.survived_score == 0.7  # Best tree is kept
        assert (
            mock_validationTree.call_count == 3
        )  # Called for each tree + one more for the final check

        # Scenario 3: Tree fails, threshold decreases
        rf_th_decease = RandomForestMC(
            th_start=0.8, delta_th=0.1, max_discard_trees=1, get_best_tree=False
        )
        rf_th_decease.feature_cols = ["f1", "f2"]
        mock_split_train_val.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_sampleFeats.return_value = ["f1"]
        mock_plantTree.side_effect = [
            create_mock_tree(survived_score=0.6),  # Fails
            create_mock_tree(survived_score=0.7),  # Fails, new threshold 0.7
            create_mock_tree(survived_score=0.75),  # Survives with new threshold
        ]
        mock_validationTree.side_effect = [0.6, 0.7, 0.75]

        with patch("src.random_forest_mc.model.log") as mock_log:
            tree_th_decease = rf_th_decease.survivedTree()
            assert tree_th_decease.survived_score == 0.75
            mock_log.info.assert_called_with("New threshold for drop: 0.7000")

    @patch("src.random_forest_mc.model.RandomForestMC.survivedTree")
    def test_fit(self, mock_survivedTree):
        rf = RandomForestMC(n_trees=2)
        df = pd.DataFrame({"f1": [1, 2], "target": ["A", "B"]})
        rf.process_dataset(df)

        mock_tree1 = create_mock_tree(survived_score=0.8)
        mock_tree2 = create_mock_tree(survived_score=0.9)
        mock_survivedTree.side_effect = [mock_tree1, mock_tree2]

        rf.fit()
        assert len(rf.data) == 2
        assert rf.survived_scores == [0.8, 0.9]
        mock_survivedTree.call_count == 2

        with pytest.raises(DatasetNotFound):
            rf_no_dataset = RandomForestMC()
            rf_no_dataset.fit()

    @patch("src.random_forest_mc.model.process_map")
    def test_fitParallel(self, mock_process_map):
        rf = RandomForestMC(n_trees=2)
        df = pd.DataFrame({"f1": [1, 2], "target": ["A", "B"]})
        rf.process_dataset(df)

        mock_tree1 = create_mock_tree(survived_score=0.8)
        mock_tree2 = create_mock_tree(survived_score=0.9)
        mock_process_map.return_value = [mock_tree1, mock_tree2]

        rf.fitParallel()
        assert len(rf.data) == 2
        assert rf.survived_scores == [0.8, 0.9]
        mock_process_map.assert_called_once()

        with pytest.raises(DatasetNotFound):
            rf_no_dataset = RandomForestMC()
            rf_no_dataset.fitParallel()

    def test_sampleClass2trees(self):
        rf = RandomForestMC()
        rf.class_vals = ["A", "B"]
        rf.data = [
            create_mock_tree(survived_score=0.8, class_vals=["A", "B"]),
            create_mock_tree(survived_score=0.7, class_vals=["A", "B"]),
        ]
        # Mock the __call__ method of the trees to return specific predictions
        rf.data[0].__call__.return_value = {"A": 0.9, "B": 0.1}  # Predicts A
        rf.data[1].__call__.return_value = {"A": 0.1, "B": 0.9}  # Predicts B

        row = pd.Series({"f1": 1})
        trees_for_A = rf.sampleClass2trees(row, "A")
        assert len(trees_for_A) == 1
        assert trees_for_A[0].survived_score == 0.8

        trees_for_B = rf.sampleClass2trees(row, "B")
        assert len(trees_for_B) == 1
        assert trees_for_B[0].survived_score == 0.7

    def test_trees2depths(self):
        rf = RandomForestMC()
        mock_tree1 = create_mock_tree()
        mock_tree1.depths = [1, 2]
        mock_tree2 = create_mock_tree()
        mock_tree2.depths = [1, 3, 4]
        rf.data = [mock_tree1, mock_tree2]
        assert rf.trees2depths == [[1, 2], [1, 3, 4]]

    def test_tree2feats(self):
        rf = RandomForestMC()
        rf.feature_cols = ["f1", "f2", "f3"]
        mock_tree = create_mock_tree()
        # Simulate the string representation of a tree containing feature names
        mock_tree.__str__.return_value = "{'f1': {'split': ...}, 'f3': {'split': ...}}"
        mock_tree.used_features = ["f1", "f3"]

        feats = rf.tree2feats(mock_tree)
        assert sorted(feats) == sorted(["f1", "f3"])

    def test_featCount(self):
        rf = RandomForestMC()
        mock_tree1 = create_mock_tree(used_features=["f1", "f2"])
        mock_tree2 = create_mock_tree(used_features=["f1"])
        rf.data = [mock_tree1, mock_tree2]

        (mean, std, min_val, max_val), counts = rf.featCount()
        assert mean == 1.5
        assert std == 0.5
        assert min_val == 1
        assert max_val == 2
        assert counts == [2, 1]

    def test_sampleClassFeatCount(self):
        rf = RandomForestMC()
        rf.class_vals = ["A", "B"]
        rf.data = [
            create_mock_tree(
                survived_score=0.8, class_vals=["A", "B"], used_features=["f1", "f2"]
            ),
            create_mock_tree(
                survived_score=0.7, class_vals=["A", "B"], used_features=["f1"]
            ),
        ]
        rf.data[0].__call__.return_value = {"A": 0.9, "B": 0.1}  # Predicts A
        rf.data[1].__call__.return_value = {"A": 0.1, "B": 0.9}  # Predicts B

        row = pd.Series({"f1": 1})
        (mean, std, min_val, max_val), counts = rf.sampleClassFeatCount(row, "A")
        assert mean == 2.0
        assert std == 0.0
        assert min_val == 2
        assert max_val == 2
        assert counts == [2]

    def test_featImportance(self):
        rf = RandomForestMC()
        mock_tree1 = create_mock_tree(used_features=["f1", "f2"])
        mock_tree2 = create_mock_tree(used_features=["f1"])
        rf.data = [mock_tree1, mock_tree2]

        importance = rf.featImportance()
        assert importance["f1"] == 1.0  # Used in both trees
        assert importance["f2"] == 0.5  # Used in one tree

    def test_sampleClassFeatImportance(self):
        rf = RandomForestMC()
        rf.class_vals = ["A", "B"]
        rf.data = [
            create_mock_tree(
                survived_score=0.8, class_vals=["A", "B"], used_features=["f1", "f2"]
            ),
            create_mock_tree(
                survived_score=0.7, class_vals=["A", "B"], used_features=["f1"]
            ),
        ]
        rf.data[0].__call__.return_value = {"A": 0.9, "B": 0.1}  # Predicts A
        rf.data[1].__call__.return_value = {"A": 0.1, "B": 0.9}  # Predicts B

        row = pd.Series({"f1": 1})
        importance = rf.sampleClassFeatImportance(row, "A")
        assert importance["f1"] == 1.0
        assert importance["f2"] == 1.0

    def test_featScoreMean(self):
        rf = RandomForestMC()
        mock_tree1 = create_mock_tree(survived_score=0.8, used_features=["f1", "f2"])
        mock_tree2 = create_mock_tree(survived_score=0.9, used_features=["f1"])
        rf.data = [mock_tree1, mock_tree2]
        rf.survived_scores = [0.8, 0.9]

        scores = rf.featScoreMean()
        assert scores["f1"] == pytest.approx((0.8 + 0.9) / 2)
        assert scores["f2"] == pytest.approx(0.8)

    def test_sampleClassFeatScoreMean(self):
        rf = RandomForestMC()
        rf.class_vals = ["A", "B"]
        rf.data = [
            create_mock_tree(
                survived_score=0.8, class_vals=["A", "B"], used_features=["f1", "f2"]
            ),
            create_mock_tree(
                survived_score=0.7, class_vals=["A", "B"], used_features=["f1"]
            ),
        ]
        rf.survived_scores = [0.8, 0.7]
        rf.data[0].__call__.return_value = {"A": 0.9, "B": 0.1}  # Predicts A
        rf.data[1].__call__.return_value = {"A": 0.1, "B": 0.9}  # Predicts B

        row = pd.Series({"f1": 1})
        scores = rf.sampleClassFeatScoreMean(row, "A")
        assert scores["f1"] == pytest.approx(0.8)
        assert scores["f2"] == pytest.approx(0.8)

    def test_featPairImportance(self):
        rf = RandomForestMC()
        rf.feature_cols = ["f1", "f2", "f3"]
        mock_tree1 = create_mock_tree(used_features=["f1", "f2"])
        mock_tree2 = create_mock_tree(used_features=["f1", "f3"])
        rf.data = [mock_tree1, mock_tree2]

        importance = rf.featPairImportance(disable_progress_bar=True)
        assert importance[("f1", "f2")] == 0.5  # Used in tree1
        assert importance[("f1", "f3")] == 0.5  # Used in tree2
        assert ("f2", "f3") not in importance

    def test_sampleClassFeatPairImportance(self):
        rf = RandomForestMC()
        rf.class_vals = ["A", "B"]
        rf.feature_cols = ["f1", "f2", "f3"]
        rf.data = [
            create_mock_tree(
                survived_score=0.8, class_vals=["A", "B"], used_features=["f1", "f2"]
            ),
            create_mock_tree(
                survived_score=0.7, class_vals=["A", "B"], used_features=["f1", "f3"]
            ),
        ]
        rf.data[0].__call__.return_value = {"A": 0.9, "B": 0.1}  # Predicts A
        rf.data[1].__call__.return_value = {"A": 0.1, "B": 0.9}  # Predicts B

        row = pd.Series({"f1": 1})
        importance = rf.sampleClassFeatPairImportance(row, "A")
        assert importance[("f1", "f2")] == 1.0
        assert ("f1", "f3") not in importance

    def test_featCorrDataFrame(self):
        rf = RandomForestMC()
        rf.feature_cols = ["f1", "f2", "f3"]
        mock_tree1 = create_mock_tree(used_features=["f1", "f2"])
        mock_tree2 = create_mock_tree(used_features=["f1", "f3"])
        rf.data = [mock_tree1, mock_tree2]

        corr_df = rf.featCorrDataFrame()
        assert isinstance(corr_df, pd.DataFrame)
        assert corr_df.index.tolist() == ["f1", "f2", "f3"]
        assert corr_df.columns.tolist() == ["f1", "f2", "f3"]
        assert corr_df.loc["f1", "f1"] == 1.0  # Importance of f1
        assert corr_df.loc["f2", "f2"] == 0.5  # Importance of f2
        assert corr_df.loc["f1", "f2"] == 0.5  # Pair importance of (f1, f2)
        assert corr_df.loc["f2", "f1"] == 0.5  # Symmetric

    def test_sampleClassFeatCorrDataFrame(self):
        rf = RandomForestMC()
        rf.class_vals = ["A", "B"]
        rf.feature_cols = ["f1", "f2", "f3"]
        rf.data = [
            create_mock_tree(
                survived_score=0.8, class_vals=["A", "B"], used_features=["f1", "f2"]
            ),
            create_mock_tree(
                survived_score=0.7, class_vals=["A", "B"], used_features=["f1", "f3"]
            ),
        ]
        rf.data[0].__call__.return_value = {"A": 0.9, "B": 0.1}  # Predicts A
        rf.data[1].__call__.return_value = {"A": 0.1, "B": 0.9}  # Predicts B

        row = pd.Series({"f1": 1})
        corr_df = rf.sampleClassFeatCorrDataFrame(row, "A")
        assert corr_df.loc["f1", "f1"] == 1.0
        assert corr_df.loc["f2", "f2"] == 1.0
        assert corr_df.loc["f1", "f2"] == 1.0
        assert corr_df.loc["f1", "f3"] == 0.0  # f3 not used in tree for class A

    def test_fill_row_missing(self):
        rf = RandomForestMC()
        row = pd.Series({"f1": 1, "f2": np.nan, "f3": 3})
        dict_values = {"f2": ["A", "B"]}
        filled_df = rf._fill_row_missing(row, dict_values)
        assert filled_df.shape == (2, 3)
        assert filled_df.loc[0, "f2"] == "A"
        assert filled_df.loc[1, "f2"] == "B"

        # No missing values
        row_no_nan = pd.Series({"f1": 1, "f2": 2, "f3": 3})
        filled_df_no_nan = rf._fill_row_missing(row_no_nan, dict_values)
        assert filled_df_no_nan is None

    def test_validationMissingValues(self):
        rf = RandomForestMC()
        rf.data = [
            create_mock_tree(used_features=["f1", "f2"]),
            create_mock_tree(used_features=["f3"]),
        ]
        rf.feature_cols = ["f1", "f2", "f3", "f4"]

        # All features in dict_values are in used_features
        dict_values_valid = {"f1": [1], "f2": [2]}
        rf._validationMissingValues(dict_values_valid)  # Should not raise error

        # Some features in dict_values are not in used_features
        dict_values_partial = {"f1": [1], "f4": [4]}
        with patch("src.random_forest_mc.model.log") as mock_log:
            rf._validationMissingValues(dict_values_partial)
            mock_log.warning.assert_called_with(
                "The Forest model have not the following feature(s): [f4]."
            )

        # All features in dict_values are not in used_features
        dict_values_invalid = {"f5": [5], "f6": [6]}
        with pytest.raises(DictValuesAllFeaturesMissing):
            rf._validationMissingValues(dict_values_invalid)

    @patch.object(RandomForestMC, "_fill_row_missing")
    def test_genFilledDataMissing(self, mock_fill_row_missing):
        rf = RandomForestMC()
        mock_fill_row_missing.side_effect = [
            pd.DataFrame({"f1": [1, 2], "f2": ["A", "B"]}),  # For first row
            None,  # For second row (no missing)
            pd.DataFrame({"f1": [3, 4], "f2": ["C", "D"]}),  # For third row
        ]

        # Test with PandasSeriesRow
        row = pd.Series({"f1": 1, "f2": np.nan})
        dict_values = {"f2": ["A", "B"]}
        original_row_df, filled_df = rf._genFilledDataMissing(row, dict_values)
        assert original_row_df.shape == (1, 2)
        assert filled_df.shape == (2, 2)

        # Test with pd.DataFrame
        df = pd.DataFrame({"f1": [1, 5, 3], "f2": [np.nan, 6, np.nan]})
        original_df, filled_df_multi = rf._genFilledDataMissing(df, dict_values)
        assert original_df.shape == (3, 2)
        assert filled_df_multi.shape == (4, 2)  # 2 from first row, 2 from third row

        # Test MissingValuesNotFound
        mock_fill_row_missing.side_effect = [
            None,
            None,
        ]  # All rows have no missing values
        with pytest.raises(MissingValuesNotFound):
            rf._genFilledDataMissing(df, dict_values)

    @patch.object(RandomForestMC, "_validationMissingValues")
    @patch.object(RandomForestMC, "_genFilledDataMissing")
    @patch.object(RandomForestMC, "predict_proba")
    def test_predictMissingValues(
        self,
        mock_predict_proba,
        mock_genFilledDataMissing,
        mock_validationMissingValues,
    ):
        rf = RandomForestMC()
        rf.class_vals = ["classA", "classB"]

        # Mock _genFilledDataMissing output
        original_data = pd.DataFrame({"f1": [1, 2], "f2": [np.nan, 4]})
        filled_data = pd.DataFrame(
            {
                "f1": [1, 1, 2, 2],
                "f2": ["A", "B", 4, 4],
                "_target_prob_classA": [0.9, 0.1, 0.8, 0.2],
                "_target_prob_classB": [0.1, 0.9, 0.2, 0.8],
            }
        )
        mock_genFilledDataMissing.return_value = (
            original_data,
            filled_data.drop(columns=["_target_prob_classA", "_target_prob_classB"]),
        )

        # Mock predict_proba output
        mock_predict_proba.return_value = [
            {"classA": 0.9, "classB": 0.1},
            {"classA": 0.1, "classB": 0.9},
            {"classA": 0.8, "classB": 0.2},
            {"classA": 0.2, "classB": 0.8},
        ]

        dict_values = {"f2": ["A", "B"]}
        result_df = rf.predictMissingValues(original_data, dict_values)

        mock_validationMissingValues.assert_called_once_with(dict_values)
        mock_genFilledDataMissing.assert_called_once_with(original_data, dict_values)
        mock_predict_proba.assert_called_once()

        assert result_df.shape[0] == 4  # 2 original rows * 2 filled options for f2
        assert "row_id" in result_df.columns
        assert result_df.loc[0, "row_id"] == 0
        assert result_df.loc[1, "row_id"] == 0
        assert result_df.loc[2, "row_id"] == 1
        assert result_df.loc[3, "row_id"] == 1

        assert result_df.loc[0, "classA"] == pytest.approx(0.9)
        assert result_df.loc[1, "classA"] == pytest.approx(0.1)


# Test custom exceptions
def test_MissingValuesNotFound_exception():
    with pytest.raises(MissingValuesNotFound) as excinfo:
        raise MissingValuesNotFound()
    assert "Dataset or row without missing values!" in str(excinfo.value)


def test_DatasetNotFound_exception():
    with pytest.raises(DatasetNotFound) as excinfo:
        raise DatasetNotFound()
    assert "Dataset not found!" in str(excinfo.value)


def test_DictValuesAllFeaturesMissing_exception():
    with pytest.raises(DictValuesAllFeaturesMissing) as excinfo:
        raise DictValuesAllFeaturesMissing()
    assert "All features in the given dictionary 'dict_values' are not found" in str(
        excinfo.value
    )
