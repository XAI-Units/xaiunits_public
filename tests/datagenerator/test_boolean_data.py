import pytest
from xaiunits.datagenerator import BooleanDataset
import torch
import sympy
from sympy.abc import A, B, C
from sympy.logic.boolalg import And, Or, Not, Xor


@pytest.fixture
def datasample():
    f = sympy.symbols("x") & sympy.symbols("y")
    data = BooleanDataset(formula=f)
    return data


@pytest.fixture
def sample_formula():
    return sympy.symbols("x") & sympy.symbols("y")


class TestBooleanData:
    def test_return_types(self, datasample):
        """Tests that both samples and labels are Tensor."""
        samples, labels = datasample._initialize_samples_labels(10)
        assert isinstance(samples, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_matching_sizes(self, datasample):
        """Tests the size of the labels and samples."""
        n_samples = 10
        samples, labels = datasample._initialize_samples_labels(n_samples)
        assert len(samples) == n_samples
        assert len(labels) == n_samples

    def test_atoms_extraction(self, sample_formula):
        """Tests the atoms of the Dataset."""
        dataset = BooleanDataset(formula=sample_formula, seed=42, n_samples=10)
        expected_atoms = tuple(sample_formula.atoms())
        assert dataset.atoms == expected_atoms

    def test_boolean_to_numeric_transformation(self):
        """Tests the conversion of boolean data to numeric."""
        formula = And(A, B)
        dataset = BooleanDataset(formula=formula, n_samples=4)
        # Check that no values other than 1.0 or -1.0 exist in samples and labels
        assert set(torch.unique(dataset.samples).tolist()) <= {
            1.0,
            -1.0,
        }, "Invalid numeric values in samples"
        assert set(torch.unique(dataset.labels).tolist()) <= {
            1.0,
            -1.0,
        }, "Invalid numeric values in labels"


class TestTruthTables:
    def test_truth_table_generation(self):
        """Tests the generation of truth tables produce correct samples and labels."""
        formula = And(A, B)
        dataset = BooleanDataset(formula=formula, n_samples=4)
        expected_samples = torch.tensor(
            [[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]]
        )
        expected_labels = torch.tensor([1.0, -1.0, -1.0, -1.0])
        assert torch.equal(
            dataset.samples, expected_samples
        ), "Samples do not match expected truth table"
        assert torch.equal(
            dataset.labels, expected_labels
        ), "Labels do not match expected truth table outcomes"

    def test_truth_table_completeness(self):
        """Tests the completeness of generated truth table."""
        formula = And(A, B)  # Simple AND operation
        num_combinations = 2 ** len(formula.atoms())  # Total combinations for 2 atoms
        dataset = BooleanDataset(formula=formula, n_samples=num_combinations)
        unique_samples = {tuple(sample.tolist()) for sample in dataset.samples}
        assert (
            len(unique_samples) == num_combinations
        ), "Not all combinations are present in the dataset"


def test_handling_of_duplicates():
    """
    Tests that when n_samples exceeds the size of the truth table,
    the dataset appropriately includes duplicates without any errors.
    """
    formula = And(A, B)
    num_combinations = 2 ** len(formula.atoms())
    n_samples = (
        num_combinations * 2
    )  # Intentionally requesting more samples than combinations
    dataset = BooleanDataset(formula=formula, n_samples=n_samples)
    assert (
        len(dataset.samples) == n_samples
    ), "The dataset does not correctly handle duplicate entries"


def test_seed_reproducibility():
    """Tests the reproducibility of dataset with the same seed."""
    formula = And(A, B)
    seed = 42
    dataset1 = BooleanDataset(formula=formula, seed=seed, n_samples=10)
    dataset2 = BooleanDataset(formula=formula, seed=seed, n_samples=10)
    assert torch.equal(
        dataset1.samples, dataset2.samples
    ), "Samples are not reproducible with the same seed"
    assert torch.equal(
        dataset1.labels, dataset2.labels
    ), "Labels are not reproducible with the same seed"


def test_edge_cases_in_sampling():
    """Tests edge cases in truth table sampling."""
    formula = And(A, B)
    # Test with one sample
    zero_sample_dataset = BooleanDataset(formula=formula, n_samples=1)
    assert (
        len(zero_sample_dataset.samples) == 1 and len(zero_sample_dataset.labels) == 1
    ), "One sample case not handled correctly"

    # Test with a very large number of samples
    large_sample_dataset = BooleanDataset(
        formula=formula, n_samples=10000
    )  # Arbitrary large number
    assert (
        len(large_sample_dataset.samples) == 10000
    ), "Large sample case not handled correctly"


def test_xor_operation():
    """Tests the correctness of the dataset generated for the XOR operation."""
    formula = Xor(A, B)
    dataset = BooleanDataset(formula=formula, n_samples=4)
    # A XOR B is True when A != B
    expected_samples = torch.tensor(
        [[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]]
    )
    expected_labels = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    assert torch.equal(dataset.samples, expected_samples), "XOR samples are incorrect"
    assert torch.equal(dataset.labels, expected_labels), "XOR labels are incorrect"


def test_nested_operators():
    """
    Tests the correctness of the dataset generated for a more complicated
    formula with nested operators.
    """
    formula = Or(And(A, Not(B)), And(Not(A), C))
    expected_samples = [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
    ]
    expected_labels = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    dataset = BooleanDataset(formula=formula, n_samples=8)
    samples_ls = [tuple(ls.tolist()) for ls in dataset.samples]
    assert sorted(expected_samples) == sorted(
        samples_ls
    ), "Nested operator samples are incorrect"
    labels_ls = dataset.labels.tolist()
    assert sorted(expected_labels) == sorted(
        labels_ls
    ), "Nested operator labels are incorrect"


if __name__ == "__main__":

    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
