import pytest
import pandas as pd
from collections import UserList
from unittest.mock import MagicMock, patch
from src.random_forest_mc.forest import BaseRandomForestMC
from src.random_forest_mc.tree import DecisionTreeMC


# Helper function to create a dummy DecisionTreeMC
def create_dummy_decision_tree(
    survived_score=0.5, data=None, class_vals=None, features=None, used_features=None
):
    if data is None:
        data = {"leaf": {"classA": 1.0}}
    if class_vals is None:
        class_vals = ["classA"]
    if features is None:
        features = ["feature1"]
    if used_features is None:
        used_features = ["feature1"]
    tree = DecisionTreeMC(data, class_vals, survived_score, features, used_features)
    # Mock the __call__ method for easier testing of useForest
    tree.__call__ = MagicMock(return_value={"classA": 1.0, "classB": 0.0})
    _ = tree.md5hexdigest
    return tree


class TestBaseRandomForestMC:
    def test_init(self):
        forest = BaseRandomForestMC()
        assert forest.n_trees == 16
        assert forest.target_col == "target"
        assert forest.min_feature is None
        assert forest.max_feature is None
        assert forest.temporal_features is False
        assert forest.soft_voting is False
        assert forest.weighted_tree is False
        assert isinstance(forest.data, UserList)
        assert forest.survived_scores == []

        forest_custom = BaseRandomForestMC(
            n_trees=5,
            target_col="label",
            min_feature=2,
            max_feature=10,
            temporal_features=True,
        )
        assert forest_custom.n_trees == 5
        assert forest_custom.target_col == "label"
        assert forest_custom.min_feature == 2
        assert forest_custom.max_feature == 10
        assert forest_custom.temporal_features is True

    def test_repr(self):
        forest = BaseRandomForestMC(n_trees=10)
        assert "BaseRandomForestMC(len(Forest)=0,n_trees=10" in repr(forest)

    def test_eq(self):
        forest1 = BaseRandomForestMC(n_trees=10)
        forest2 = BaseRandomForestMC(n_trees=10)
        forest3 = BaseRandomForestMC(n_trees=5)

        assert forest1 == forest2
        assert forest1 != forest3

        with pytest.raises(TypeError):
            _ = forest1 == "not_a_forest"

    @pytest.fixture
    def sample_forest(self):
        forest = BaseRandomForestMC(n_trees=2)
        forest.class_vals = ["classA", "classB"]
        forest.data = [
            create_dummy_decision_tree(
                survived_score=0.8,
                data={
                    "f1": {
                        "split": {
                            "feat_type": "numeric",
                            "split_val": 0.5,
                            ">=": {"leaf": {"classA": 0.9, "classB": 0.1}},
                            "<": {"leaf": {"classA": 0.2, "classB": 0.8}},
                        }
                    }
                },
            ),
            create_dummy_decision_tree(
                survived_score=0.7,
                data={
                    "f2": {
                        "split": {
                            "feat_type": "categorical",
                            "split_val": "X",
                            ">=": {"leaf": {"classA": 0.3, "classB": 0.7}},
                            "<": {"leaf": {"classA": 0.6, "classB": 0.4}},
                        }
                    }
                },
            ),
        ]
        forest.survived_scores = [0.8, 0.7]
        forest.feature_cols = ["f1", "f2"]
        # Mock the __call__ method for the dummy trees to return specific probabilities
        forest.data[0].__call__.return_value = {"classA": 0.9, "classB": 0.1}
        forest.data[1].__call__.return_value = {"classA": 0.3, "classB": 0.7}
        return forest

    def test_predict_proba_and_predict(self, sample_forest):
        # Test with PandasSeriesRow
        row = pd.Series({"f1": 0.6, "f2": "Y"})
        result_proba = sample_forest.predict_proba(row)
        assert isinstance(result_proba, dict)
        # The mocked tree calls will return the same value, so the useForest will average them
        # (0.9 + 0.3) / 2 = 0.6 for classA
        # (0.1 + 0.7) / 2 = 0.4 for classB
        assert result_proba["classA"] == pytest.approx(0.6)
        assert result_proba["classB"] == pytest.approx(0.4)

        # Test with pd.DataFrame (prob_output=True)
        df = pd.DataFrame([{"f1": 0.6, "f2": "Y"}, {"f1": 0.1, "f2": "X"}])
        results_proba_df = sample_forest.predict_proba(df)
        assert isinstance(results_proba_df, list)
        assert len(results_proba_df) == 2
        assert isinstance(results_proba_df[0], dict)

        # Test with pd.DataFrame (prob_output=False)
        results_class_df = sample_forest.predict(df, prob_output=False)
        assert isinstance(results_class_df, list)
        assert len(results_class_df) == 2
        assert results_class_df[0] == "classA"  # Based on maxProbClas of 0.6 for classA

        with pytest.raises(TypeError):
            sample_forest.predict("not_a_series_or_df")

    def test_mergeForest(self):
        forest1 = BaseRandomForestMC(n_trees=2)
        forest1.class_vals = ["A", "B"]
        forest1.feature_cols = ["x", "y"]
        forest1.data = [
            create_dummy_decision_tree(
                survived_score=0.1, class_vals=["A", "B"], features=["x", "y"]
            )
        ]
        forest1.survived_scores = [0.1]

        forest2 = BaseRandomForestMC(n_trees=2)
        forest2.class_vals = ["A", "B"]
        forest2.feature_cols = ["x", "y"]
        forest2.data = [
            create_dummy_decision_tree(
                survived_score=0.2, class_vals=["A", "B"], features=["x", "y"]
            )
        ]
        forest2.survived_scores = [0.2]

        # Test merge by add
        forest1_copy = BaseRandomForestMC(n_trees=2)
        forest1_copy.class_vals = ["A", "B"]
        forest1_copy.feature_cols = ["x", "y"]
        forest1_copy.data = [
            create_dummy_decision_tree(
                survived_score=0.1, class_vals=["A", "B"], features=["x", "y"]
            )
        ]
        forest1_copy.survived_scores = [0.1]
        forest1_copy.mergeForest(forest2, by="add")
        assert len(forest1_copy.data) == 2
        assert forest1_copy.survived_scores == [0.1, 0.2]

        # Test merge by score (N=1)
        forest1_copy = BaseRandomForestMC(n_trees=2)
        forest1_copy.class_vals = ["A", "B"]
        forest1_copy.feature_cols = ["x", "y"]
        forest1_copy.data = [
            create_dummy_decision_tree(
                survived_score=0.1, class_vals=["A", "B"], features=["x", "y"]
            )
        ]
        forest1_copy.survived_scores = [0.1]
        forest1_copy.mergeForest(forest2, N=1, by="score")
        assert len(forest1_copy.data) == 1
        assert forest1_copy.survived_scores == [
            0.2
        ]  # Only the tree with score 0.2 should remain

        # Test merge by random (N=1) - difficult to test deterministically, so just check length
        forest1_copy = BaseRandomForestMC(n_trees=2)
        forest1_copy.class_vals = ["A", "B"]
        forest1_copy.feature_cols = ["x", "y"]
        forest1_copy.data = [
            create_dummy_decision_tree(
                survived_score=0.1, class_vals=["A", "B"], features=["x", "y"]
            )
        ]
        forest1_copy.survived_scores = [0.1]
        forest1_copy.mergeForest(forest2, N=1, by="random")
        assert len(forest1_copy.data) == 1

        with pytest.raises(TypeError):
            forest1.mergeForest("not_a_forest")

        # Test ValueError for different classes/features
        forest_diff_class = BaseRandomForestMC(n_trees=1)
        forest_diff_class.class_vals = ["C", "D"]
        forest_diff_class.feature_cols = ["x", "y"]
        with pytest.raises(ValueError):
            forest1.mergeForest(forest_diff_class)

        forest_diff_features = BaseRandomForestMC(n_trees=1)
        forest_diff_features.class_vals = ["A", "B"]
        forest_diff_features.feature_cols = ["a", "b"]
        with pytest.raises(ValueError):
            forest1.mergeForest(forest_diff_features)

    def test_validFeaturesTemporal(self):
        forest = BaseRandomForestMC()
        forest.feature_cols = ["feat_1", "feat_2", "feat_3"]
        assert forest.validFeaturesTemporal() is True

        forest.feature_cols = ["feat_1", "feat_abc", "feat_3"]
        assert forest.validFeaturesTemporal() is False

        forest.feature_cols = []
        assert (
            forest.validFeaturesTemporal() is True
        )  # All elements satisfy the condition vacuously

    def test_setSoftVoting_and_setWeightedTrees(self):
        forest = BaseRandomForestMC()
        assert forest.soft_voting is False
        assert forest.weighted_tree is False

        forest.setSoftVoting(True)
        assert forest.soft_voting is True

        forest.setWeightedTrees(True)
        assert forest.weighted_tree is True

        forest.setSoftVoting(False)
        assert forest.soft_voting is False

        forest.setWeightedTrees(False)
        assert forest.weighted_tree is False

    def test_reset_forest(self):
        forest = BaseRandomForestMC()
        forest.data = [create_dummy_decision_tree()]
        forest.survived_scores = [0.5]
        forest.reset_forest()
        assert forest.data == []
        assert forest.survived_scores == []

    def test_model2dict_and_dict2model(self):
        forest = BaseRandomForestMC(n_trees=1, min_feature=1, max_feature=1)
        forest.class_vals = ["A", "B"]
        forest.numeric_cols = ["x"]
        forest.feature_cols = ["x", "y"]
        forest.type_of_cols = {"x": "numeric", "y": "categorical"}
        forest.data = [
            create_dummy_decision_tree(
                survived_score=0.9,
                class_vals=["A", "B"],
                features=["x", "y"],
                used_features=["x"],
            )
        ]
        forest.survived_scores = [0.9]

        model_dict = forest.model2dict()
        assert isinstance(model_dict, dict)
        assert "Forest" in model_dict
        assert len(model_dict["Forest"]) == 1
        assert model_dict["min_feature"] == 1
        assert model_dict["survived_scores"] == [0.9]

        new_forest = BaseRandomForestMC()
        new_forest.dict2model(model_dict)
        assert (
            new_forest.n_trees == forest.n_trees
        )  # n_trees is not saved in attr_to_save, so it will be default
        assert new_forest.min_feature == forest.min_feature
        assert new_forest.max_feature == forest.max_feature
        assert new_forest.class_vals == forest.class_vals
        assert new_forest.survived_scores == forest.survived_scores
        assert len(new_forest.data) == len(forest.data)
        assert new_forest.data[0].survived_score == forest.data[0].survived_score

        # Test add=True
        forest_to_add = BaseRandomForestMC(n_trees=1)
        forest_to_add.class_vals = ["A", "B"]
        forest_to_add.numeric_cols = ["x"]
        forest_to_add.feature_cols = ["x", "y"]
        forest_to_add.type_of_cols = {"x": "numeric", "y": "categorical"}
        forest_to_add.data = [
            create_dummy_decision_tree(
                survived_score=0.5,
                class_vals=["A", "B"],
                features=["x", "y"],
                used_features=["y"],
            )
        ]
        forest_to_add.survived_scores = [0.5]
        model_dict_to_add = forest_to_add.model2dict()

        new_forest_add = BaseRandomForestMC()
        new_forest_add.dict2model(model_dict)
        new_forest_add.dict2model(model_dict_to_add, add=True)
        assert len(new_forest_add.data) == 2
        assert new_forest_add.survived_scores == [0.9, 0.5]

    def test_drop_duplicated_trees(self):
        tree1 = create_dummy_decision_tree(survived_score=0.1)
        tree2 = create_dummy_decision_tree(survived_score=0.2)  # Same hash as tree1
        tree3 = create_dummy_decision_tree(
            survived_score=0.3,
            data={"leaf": {"classB": 1.0}},
            used_features=["feature2"],
        )  # Different hash

        forest = BaseRandomForestMC()
        forest.data = [tree1, tree2, tree3]
        forest.survived_scores = [0.1, 0.2, 0.3]

        initial_len = len(forest.data)
        dropped_count = forest.drop_duplicated_trees()

        assert len(forest.data) == initial_len - 1  # One duplicate removed
        assert dropped_count == 2  # Number of unique trees remaining
        assert forest.data == [tree1, tree3]  # tree2 should be removed
        assert forest.survived_scores == [0.1, 0.3]

    def test_Forest_properties(self):
        forest = BaseRandomForestMC()
        assert forest.Forest_size == 0
        assert forest.Forest == []

        forest.data = [create_dummy_decision_tree()]
        assert forest.Forest_size == 1
        assert len(forest.Forest) == 1

    def test_maxProbClas(self):
        leaf1 = {"classA": 0.8, "classB": 0.2}
        assert BaseRandomForestMC.maxProbClas(leaf1) == "classA"

        leaf2 = {"classA": 0.1, "classB": 0.9}
        assert BaseRandomForestMC.maxProbClas(leaf2) == "classB"

        leaf3 = {"classA": 0.5, "classB": 0.5}
        assert (
            BaseRandomForestMC.maxProbClas(leaf3) == "classA"
        )  # First one in sorted order

    def test_useForest_soft_voting_weighted_tree(self, sample_forest):
        sample_forest.setSoftVoting(True)
        sample_forest.setWeightedTrees(True)
        row = pd.Series({"f1": 0.6, "f2": "Y"})

        # Mocked tree calls:
        # tree1 (score 0.8): {"classA": 0.9, "classB": 0.1}
        # tree2 (score 0.7): {"classA": 0.3, "classB": 0.7}

        # classA: (0.9 * 0.8) + (0.3 * 0.7) = 0.72 + 0.21 = 0.93
        # classB: (0.1 * 0.8) + (0.7 * 0.7) = 0.08 + 0.49 = 0.57
        # Total score: 0.8 + 0.7 = 1.5
        # Normalized classA: 0.93 / 1.5 = 0.62
        # Normalized classB: 0.57 / 1.5 = 0.38

        result = sample_forest.useForest(row)
        assert result["classA"] == pytest.approx(0.62)
        assert result["classB"] == pytest.approx(0.38)

    def test_useForest_soft_voting_unweighted_tree(self, sample_forest):
        sample_forest.setSoftVoting(True)
        sample_forest.setWeightedTrees(False)
        row = pd.Series({"f1": 0.6, "f2": "Y"})

        # Mocked tree calls:
        # tree1: {"classA": 0.9, "classB": 0.1}
        # tree2: {"classA": 0.3, "classB": 0.7}

        # classA: (0.9 + 0.3) / 2 = 0.6
        # classB: (0.1 + 0.7) / 2 = 0.4

        result = sample_forest.useForest(row)
        assert result["classA"] == pytest.approx(0.6)
        assert result["classB"] == pytest.approx(0.4)

    def test_useForest_hard_voting_weighted_tree(self, sample_forest):
        sample_forest.setSoftVoting(False)
        sample_forest.setWeightedTrees(True)
        row = pd.Series({"f1": 0.6, "f2": "Y"})

        # Mocked tree calls and maxProbClas:
        # tree1 (score 0.8): maxProbClas({"classA": 0.9, "classB": 0.1}) -> "classA"
        # tree2 (score 0.7): maxProbClas({"classA": 0.3, "classB": 0.7}) -> "classB"

        # classA score: 0.8
        # classB score: 0.7
        # Total score: 1.5
        # Normalized classA: 0.8 / 1.5 = 0.5333...
        # Normalized classB: 0.7 / 1.5 = 0.4666...

        result = sample_forest.useForest(row)
        assert result["classA"] == pytest.approx(0.8 / 1.5)
        assert result["classB"] == pytest.approx(0.7 / 1.5)

    def test_useForest_hard_voting_unweighted_tree(self, sample_forest):
        sample_forest.setSoftVoting(False)
        sample_forest.setWeightedTrees(False)
        row = pd.Series({"f1": 0.6, "f2": "Y"})

        # Mocked tree calls and maxProbClas:
        # tree1: maxProbClas({"classA": 0.9, "classB": 0.1}) -> "classA"
        # tree2: maxProbClas({"classA": 0.3, "classB": 0.7}) -> "classB"

        # classA count: 1
        # classB count: 1
        # Total trees: 2
        # Normalized classA: 1 / 2 = 0.5
        # Normalized classB: 1 / 2 = 0.5

        result = sample_forest.useForest(row)
        assert result["classA"] == pytest.approx(0.5)
        assert result["classB"] == pytest.approx(0.5)

    def test_testForest(self, sample_forest):
        df = pd.DataFrame([{"f1": 0.6, "f2": "Y"}, {"f1": 0.1, "f2": "X"}])
        # Based on hard voting, unweighted (default for testForest)
        sample_forest.setSoftVoting(False)
        sample_forest.setWeightedTrees(False)

        # For first row, useForest returns {"classA": 0.5, "classB": 0.5}, maxProbClas -> "classA"
        # For second row, useForest returns {"classA": 0.5, "classB": 0.5}, maxProbClas -> "classA"
        results = sample_forest.testForest(df)
        assert results == ["classA", "classA"]

    def test_testForestProbs(self, sample_forest):
        df = pd.DataFrame([{"f1": 0.6, "f2": "Y"}, {"f1": 0.1, "f2": "X"}])
        # Based on soft voting, unweighted (default for useForest in this context)
        sample_forest.setSoftVoting(True)
        sample_forest.setWeightedTrees(False)

        # For both rows, useForest returns {"classA": 0.6, "classB": 0.4}
        results = sample_forest.testForestProbs(df)
        assert len(results) == 2
        assert results[0]["classA"] == pytest.approx(0.6)
        assert results[1]["classA"] == pytest.approx(0.6)

    @patch("src.random_forest_mc.forest.process_map")
    def test_testForestParallel(self, mock_process_map, sample_forest):
        df = pd.DataFrame([{"f1": 0.6, "f2": "Y"}, {"f1": 0.1, "f2": "X"}])
        mock_process_map.return_value = ["classA", "classB"]

        results = sample_forest.testForestParallel(df, max_workers=2)
        mock_process_map.assert_called_once()
        args, kwargs = mock_process_map.call_args
        assert args[0] == sample_forest._testForest_func
        assert len(args[1]) == 2  # Two rows in df
        assert kwargs["desc"] == "Testing the forest"
        assert results == ["classA", "classB"]

    @patch("src.random_forest_mc.forest.process_map")
    def test_testForestProbsParallel(self, mock_process_map, sample_forest):
        df = pd.DataFrame([{"f1": 0.6, "f2": "Y"}, {"f1": 0.1, "f2": "X"}])
        mock_process_map.return_value = [
            {"classA": 0.6, "classB": 0.4},
            {"classA": 0.5, "classB": 0.5},
        ]

        results = sample_forest.testForestProbsParallel(df, max_workers=2)
        mock_process_map.assert_called_once()
        args, kwargs = mock_process_map.call_args
        assert args[0] == sample_forest.useForest
        assert len(args[1]) == 2  # Two rows in df
        assert kwargs["desc"] == "Testing the forest"
        assert results == [
            {"classA": 0.6, "classB": 0.4},
            {"classA": 0.5, "classB": 0.5},
        ]
