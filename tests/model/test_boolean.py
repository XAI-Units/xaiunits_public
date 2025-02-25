from itertools import product

import pytest
import torch
from sympy import Symbol, symbols
from sympy.logic.boolalg import And, Not, Or, truth_table
from xaiunits.model.boolean import *


@pytest.fixture
def boolean_model():
    all_sym = symbols("a b")
    a, b = all_sym
    formula = a | b
    model = PropFormulaNN(formula, all_sym)
    return model


class TestPropFormulaNN:
    @staticmethod
    def generate_permutations(size):
        """
        Helper method to generate all possible permutations of 1 and -1.
        """
        permutations = list(product([-1, 1], repeat=size))
        permutations = [torch.Tensor(a) for a in permutations]
        return permutations

    @staticmethod
    def evaluate_formula(formula, all_symbols):
        """
        Helper method to evaluate the given formula with all possible input permutations.
        """
        model = PropFormulaNN(formula, all_symbols)
        input = torch.stack(TestPropFormulaNN.generate_permutations(len(all_symbols)))
        output = model(input).flatten()
        return [val.item() == 1.0 for val in list(output)]

    def test_correctness(self):
        """
        Tests for correctness for a random propositional formula
        """
        all_sym = symbols("a b c d e f g")
        a, b, c, d, e, f, g = all_sym
        formula = (
            ((~a | b) & (~c | d) & (~e | f))
            | (g & a & c & b)
            | (~(g & a) & (~b | c | d) & (e | ~f))
        )
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_negation(self):
        """
        Tests for correctness for negation (NOT) operation.
        """
        a = Symbol("a")
        gt_output = [k[1] for k in truth_table(~a, [a])]
        output = self.evaluate_formula(~a, (a,))
        assert gt_output == output
        gt_output = [k[1] for k in truth_table(Not(a), [a])]
        output = self.evaluate_formula(Not(a), (a,))
        assert gt_output == output

    def test_identity(self):
        """
        Tests for correctness for identity operation.
        """
        all_sym = symbols("a")
        a = all_sym
        gt_output = [k[1] for k in truth_table(a, [a])]
        output = self.evaluate_formula(a, (a,))
        assert gt_output == output

    def test_atom_layer(self, boolean_model):
        """Test the linear Layer from atom layer method"""
        output = boolean_model._atom_layer()
        assert isinstance(output, torch.nn.Linear)
        assert output.in_features == 2
        assert output.out_features == 2

    def test_identity_weights(self, boolean_model):
        """Test the identity weights output"""
        output = boolean_model._identity_weights()
        assert torch.equal(output[0], torch.Tensor([[1.0], [-1.0]]))
        assert torch.equal(output[1], torch.Tensor([1.0, -1.0]))

    def test_post_block_arg_num_ls(self, boolean_model):
        """Test the post block argument method"""
        a = [2, 4, 9]
        assert boolean_model._post_block_arg_num_ls(a) == [1, 2, 5]

    def test_get_height_layers(self, boolean_model):
        """Test the output of the get height layers method"""
        out = boolean_model._get_height_layers(0)
        assert len(out) == 3
        assert isinstance(out[0], torch.nn.Linear)
        assert isinstance(out[-1], torch.nn.Linear)

    def test_or_weight(self, boolean_model):
        """Test the weights of the OR operation"""
        sub_inter_dim = 8
        operator = "OR"
        inter_weights, out_weights = boolean_model._and_or_weights(
            sub_inter_dim, operator
        )
        assert inter_weights.shape == torch.Size([8, 4])
        assert out_weights.shape == torch.Size([2, 8])

    # Verify the dimensions
    # self.assertEqual(inter_weights.shape, (sub_inter_dim, 2))
    # self.assertEqual(out_weights.shape, (sub_inter_dim,))

    def test_disjunction(self):
        """
        Tests for correctness for disjunction (OR) operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = a | b
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = Or(a, b)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_conjunction(self):
        """
        Tests for correctness for conjunction (AND) operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = a & b
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = And(a, b)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_xor(self):
        """
        Tests for correctness for exclusive disjunction (XOR) operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = (a & ~b) | (~a & b)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = Or(And(a, Not(b)), And(Not(a), b))
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_nand(self):
        """
        Tests for correctness for NAND operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = ~(a & b)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = Not(And(a, b))
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_nor(self):
        """
        Tests for correctness for NOR operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = ~(a | b)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = Not(Or(a, b))
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_implies(self):
        """
        Tests for correctness for implication (IMPLIES) operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = ~a | b
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = Or(Not(a), b)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_equivalence(self):
        """
        Tests for correctness for equivalence (IFF) operation.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        formula = (~a | b) & (~b | a)
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output
        formula = And(Or(Not(a), b), Or(a, Not(b)))
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output

    def test_combined_logic_operations(self):
        """
        Tests for correctness of a complex formula combining various logic operations.
        """
        all_sym = symbols("a b")
        a, b = all_sym
        # Construct a complex formula combining AND, OR, XOR, NAND, NOR, IMPLIES, and IFF
        formula = And(
            Or(And(a, b), Or(And(a, Not(b)), And(Not(a), b))),
            Not(And(Not(And(a, b)), Or(Not(a), b))),
            Not(Or(And(Or(Not(a), b), Or(a, Not(b))), Not(b))),
        )
        gt_output = [k[1] for k in truth_table(formula, all_sym)]
        output = self.evaluate_formula(formula, all_sym)
        assert gt_output == output


@pytest.fixture
def sample_parse_tree():
    a, b, c, d = symbols("a b c d")
    expr = (a & b) | (~b & c & ~d) | ~c
    return ParseTree(expr)


class TestParseTree:
    def test_parse_tree_first_layer(self, sample_parse_tree):
        """
        Tests for whether the ParseTree is correct for the first layer.
        """
        assert sample_parse_tree.node == "Or"
        assert len(sample_parse_tree.subtrees) == 3
        subtrees_ls = [subtree.node for subtree in sample_parse_tree.subtrees]
        assert "And" in subtrees_ls
        subtrees_ls.remove("And")
        assert "Id" in subtrees_ls
        subtrees_ls.remove("Id")
        assert "Id" in subtrees_ls
        subtrees_ls.remove("Id")
        assert not subtrees_ls
        assert sample_parse_tree.height == 3

    def test_initialization(self, sample_parse_tree):
        """Test the initialization of the ParseTree"""
        assert sample_parse_tree.node == "Or"

    def test_get_leafs(self, sample_parse_tree):
        """
        Tests the get_leafs method.
        """
        leafs = [leaf.node for leaf in sample_parse_tree.get_leafs()]
        assert len(leafs) == 6
        assert Symbol("a") in leafs
        leafs.remove(Symbol("a"))
        for _ in range(2):
            assert Symbol("b") in leafs
            leafs.remove(Symbol("b"))
        for _ in range(2):
            assert Symbol("c") in leafs
            leafs.remove(Symbol("c"))
        assert Symbol("d") in leafs
        leafs.remove(Symbol("d"))
        assert not leafs

    def test_get_height_subtrees(self, sample_parse_tree):
        """
        Tests the get_height_subtrees method.
        """
        assert sample_parse_tree.get_leafs() == sample_parse_tree.get_height_subtrees(0)

        subtrees_height_1 = sample_parse_tree.get_height_subtrees(1)
        assert len(subtrees_height_1) == 5
        nodes_1 = [subtree.node for subtree in subtrees_height_1]
        assert "And" in nodes_1
        nodes_1.remove("And")
        for _ in range(3):
            assert "Not" in nodes_1
            nodes_1.remove("Not")
        assert "Id" in nodes_1
        nodes_1.remove("Id")
        assert not nodes_1

        subtrees_height_2 = sample_parse_tree.get_height_subtrees(2)
        assert len(subtrees_height_2) == 3
        nodes_2 = [subtree.node for subtree in subtrees_height_2]
        assert "And" in nodes_2
        nodes_2.remove("And")
        for _ in range(2):
            assert "Id" in nodes_2
            nodes_2.remove("Id")
        assert not nodes_2

        subtrees_height_3 = sample_parse_tree.get_height_subtrees(3)
        assert len(subtrees_height_3) == 1
        assert subtrees_height_3[0] is sample_parse_tree

    def test_repr(self, sample_parse_tree):
        """
        Tests the __repr__ method.
        """
        representation = repr(sample_parse_tree)
        assert representation == "Or"
        assert representation == sample_parse_tree.node


# this block is only executed when you run the file directly as `python this_script.py`
# but you should be running `pytest` or `pytest this_script.py`
if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
