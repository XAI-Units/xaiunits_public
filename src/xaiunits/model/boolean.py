from math import ceil, log2

import torch
import torch.nn as nn
from sympy import srepr, symbols
from sympy.core.function import FunctionClass
from typing import Tuple, List


class ParseTree:
    """
    A class that parses and represents a propositional formula in a tree structure.

    Attributes:
        node (str): The node of the ParseTree.
        subtrees (list[ParseTree]): The subtrees of the ParseTree.
        height (int): The height of the ParseTree relative to the root.
    """

    def __init__(self, expr=None):
        """
        Initializes a ParseTree instance.

        If no expression is given, then all the attributes will be set to None.
        If an expression is given, the expression will be parsed, where its
        node and subtrees will be extracted accordingly. Also, the height of the ParseTree and
        its subtrees will be determined.

        If this ParseTree instance is the root of the entire expression, then the horizontal
        position of the ParseTree and its subtrees will be assigned.

        Args:
            expr (str, optional): The expression to be parsed. Defaults to None.
        """
        if expr is not None:
            self.node, self.subtrees = self._parse(expr)
            self.height = self._get_height()
            if self.subtrees is not None:
                self._fill_height_gaps()
        else:
            self.node = None
            self.subtrees = None
            self.height = None

    def get_leafs(self):
        """
        Retrieves the list of all leafs nodes in the ParseTree.

        Returns:
            list[ParseTree]: List of all ParseTree instances contianing
                leaf nodes/atomic variables of ParseTree.
        """
        return self.get_height_subtrees(0)

    def get_height_subtrees(self, target_height):
        """
        Returns the list of all subtrees with the specified height within the ParseTree.

        Args:
            target_height (int): The target height of interest.

        Returns:
            List[ParseTree]: A list of ParseTree instances with the specified height relative
                to the root.
        """
        if self.height == target_height:
            return [self]
        elif self.height < target_height:
            return []
        else:
            return [
                node
                for subtree in self.subtrees
                for node in subtree.get_height_subtrees(target_height)
            ]

    def __repr__(self):
        """
        Returns the string representation of the ParseTree.

        Returns:
            str: The string representation of the node.
        """
        return self.node

    def _get_height(self):
        """
        Returns the height of the ParseTree.

        A ParseTree with a single node and no subtree will have height 0.

        Returns:
            int: The height of the ParseTree.
        """
        if self.subtrees is None:
            return 0
        else:
            return 1 + max([subtree.height for subtree in self.subtrees])

    def _fill_height_gaps(self):
        """
        Adds trees between nodes such that the height difference between each
        node is kept at 1.
        """
        for i, subtree in enumerate(self.subtrees):
            if subtree.height < self.height - 1:
                self.subtrees[i] = self._create_identity_trees(subtree)

    def _create_identity_trees(self, subtree):
        """
        Creates ParseTrees that carry the idenity operation.

        Args:
            subtree (ParseTree): The subtree to connect with using intermediate
                identity trees.

        Returns:
            ParseTree: The first ParseTree that is connected with the target subtree
                through the chain of intermidate identity trees.
        """
        curr_tree = subtree
        for height in range(subtree.height + 1, self.height):
            dummy_tree = ParseTree()
            dummy_tree.node = "Id"
            dummy_tree.subtrees = [curr_tree]
            dummy_tree.height = height
            curr_tree = dummy_tree
        return curr_tree

    def _parse(self, expr):
        """
        Parses string expression of a propositional formula to identify its current node/operator
        and the subtrees/subformulas for its sub-expressions.

        Args:
            expr (sympy.Expr): The expression to be parsed.

        Returns:
            tuple[str, list[ParseTree]]: A tuple containing the current node and its subtrees.
        """
        # Leaf case
        if expr.is_Atom:
            return expr, None

        # Else this is a logical operator, recursive case
        node = expr.func.__name__
        if node not in ["And", "Or", "Not"]:
            raise ValueError(f"Unsupported sympy expression type: {expr}")
        subtrees = [ParseTree(arg) for arg in expr.args]
        return node, subtrees


class PropFormulaNN(nn.Sequential):
    """
    A neural network model representing a propositional formula that is constructed
    using only NOT (~), OR (|), AND (&) with syntax from SymPy.

    It takes inputs where -1 represents False and +1 represents True.

    Inherits from:
        torch.nn.Sequential: Parent class for implementing neural networks with modules defined in a
        sequential manner.

    Attributes:
        parse_tree (ParseTree): The ParseTree representation of the formula.
        atoms (tuple[str]): The unique atoms in the formula. The ordering of this
            tuple matters in determining the ordering of the input to the network.
    """

    def __init__(self, formula_expr: FunctionClass, atoms: Tuple) -> None:
        """
        Initializes a PropFormulaNN instance and generates all the torch.nn.Linear
        layers necessary to construct the neural network.

        Args:
            formula_expr (sympy.core.function.FunctionClass): The expression representing
                the propositional formula. Constructed using only syntax from SymPy with
                the operators NOT (~), OR (|), AND (&).
            atoms (tuple[str]): A tuple of unique atoms in the formula.
        """
        self.parse_tree = ParseTree(formula_expr)
        self.atoms = atoms
        super().__init__(*self._init_layers())

    def _init_layers(self) -> List[nn.Module]:
        """
        Initializes all the layers of the neural network into a list of nn.Module.

        The layers are initialized and created in ascending order of height in the
        ParseTree representation of the formula.

        Returns:
            list[torch.nn.Module]: List of neural network layers.
        """
        layers = [self._atom_layer()]
        for height in range(self.parse_tree.height):
            layers += self._get_height_layers(height)
        return layers

    def _atom_layer(self) -> torch.nn.Module:
        """
        Initializes the first layer of the neural network that duplicates and reorders
        the input propositional atoms based on their ordering in the ParseTree.

        Returns:
            torch.nn.Linear: Linear layer that duplicates and reorders the input atoms.
        """
        n_atoms = len(self.atoms)
        atom_leafs = [leaf.node for leaf in self.parse_tree.get_leafs()]
        n_atom_leafs = len(atom_leafs)

        # assign weights
        weights = torch.zeros((n_atom_leafs, n_atoms))
        for i, leaf in enumerate(atom_leafs):
            weights[i, self.atoms.index(leaf)] = 1.0

        # define layer
        layer = nn.Linear(n_atoms, n_atom_leafs, bias=False)
        layer.weight = nn.Parameter(weights)
        return layer

    def _get_height_layers(self, height: int) -> List[torch.nn.Module]:
        """
        Retrieves layers corresponding to a specific height in the ParseTree.

        Args:
            height (int): The specified height within the ParseTree representatation.

        Returns:
            list[torch.nn.Module]: List of neural network layers for the nodes that are located
                at the given height in the ParseTree representation.
        """
        subtree_ls = self.parse_tree.get_height_subtrees(height + 1)
        count_and_or = self._count_binary_operators(subtree_ls)
        if count_and_or > 0:
            return self._init_binary_operator_layers(subtree_ls)
        else:
            return self._init_unary_operator_layer(subtree_ls)

    def _init_binary_operator_layers(
        self, subtree_ls: List[ParseTree]
    ) -> List[torch.nn.Module]:
        """
        Initializes the all the layers associated with carrying out the operators
        involved in the given list of subtrees.

        It assumes that the given list of subtrees contains at least one that involves
        a binary operator/connective.

        Args:
            subtree_ls (list[ParseTree]): List of ParseTree instances with the same height in the full
                ParseTree that includes binary operators.

        Returns:
            list[torch.nn.Module]: List of neural network layers for binary operators.
        """
        arg_num_ls = [len(tree.subtrees) for tree in subtree_ls]
        num_blocks = ceil(log2(max(arg_num_ls)))
        height_layers = []
        for block in range(num_blocks):
            if block == 0:
                height_layers += self._binary_operator_block(
                    arg_num_ls, subtree_ls, first_block=True
                )
            else:
                height_layers += self._binary_operator_block(arg_num_ls, subtree_ls)
            arg_num_ls = self._post_block_arg_num_ls(arg_num_ls)
        return height_layers

    def _binary_operator_block(
        self,
        arg_num_ls: List[int],
        subtree_ls: List[ParseTree],
        first_block: bool = False,
    ) -> List[torch.nn.Module]:
        """
        Creates a block of layers that simpliefies the subexpression for the given list of subtrees
        by applying the binary operator/connective once whenever possible.

        The binary operator/connective from the same subtree can be applied simultaneously as long
        as the number of arguments inputted to it permits (i.e. divisible by two).

        Args:
            arg_num_ls (list[int]): List of number of arguments that the operator/node of each
                subtrees takes.
            subtree_ls (list[ParseTree]): List of subtrees with the same height in the ParseTree.
            first_block (bool): Indicates if this is the first block of the layers for this
                collection of subtrees with the same height.

        Returns:
            list[torch.nn.Module]: List of neural network layers in a single block.
        """
        in_dim = sum(arg_num_ls)
        inter_dim = 2 * in_dim
        out_dim = sum(self._post_block_arg_num_ls(arg_num_ls))

        # create weights
        inter_weights_ls = []
        out_weights_ls = []
        for i in range(len(arg_num_ls)):
            if arg_num_ls[i] == 1:
                inter_weights_block, out_weights_block = self._identity_weights()
                if first_block and subtree_ls[i].node == "Not":
                    out_weights_block = -1 * out_weights_block
            elif subtree_ls[i].node == "Or" or subtree_ls[i].node == "And":
                inter_weights_block, out_weights_block = self._and_or_weights(
                    2 * arg_num_ls[i], subtree_ls[i].node
                )

            # add sub-blocks
            inter_weights_ls.append(inter_weights_block)
            out_weights_ls.append(out_weights_block)

        # define layers
        inter_layer = nn.Linear(in_dim, inter_dim, bias=False)
        out_layer = nn.Linear(inter_dim, out_dim, bias=False)
        inter_layer.weight = nn.Parameter(torch.block_diag(*inter_weights_ls))
        out_layer.weight = nn.Parameter(torch.block_diag(*out_weights_ls))

        return [inter_layer, nn.ReLU(), out_layer]

    def _identity_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the weights for the two nn.Linear layers in an identity block.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Inter and out layer weights for an identity block.
        """
        # inter layer weights
        inter_weights_block = torch.ones((2, 1))
        inter_weights_block[-1] = -1.0

        # out layer weights
        out_weights_block = torch.ones(2)
        out_weights_block[-1] = -1.0
        return inter_weights_block, out_weights_block

    def _and_or_weights(
        self, sub_inter_dim: int, operator: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates weights for the two nn.Linear layers in an AND or OR block.

        Args:
            sub_inter_dim (int): Dimension of the intermediate layer.
            operator (str): The operator type (AND or OR).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Inter and out layer weights for an AND or OR block.
        """
        # inter layer weights
        active_inter_weights = torch.ones((4, 2))
        active_inter_weights[1] = -1.0
        active_inter_weights[2, 1] = -1.0
        active_inter_weights[3, 0] = -1.0
        inter_weights_ls = [active_inter_weights for _ in range(sub_inter_dim // 4)]
        if sub_inter_dim % 4 != 0:
            inactive_inter_weights = torch.ones((2, 1))
            inactive_inter_weights[-1] = -1.0
            inter_weights_ls.append(inactive_inter_weights)

        # out layer weights
        active_out_weights = torch.ones(4)
        if operator == "And":
            active_out_weights[1:] = -1.0
        else:
            active_out_weights[1] = -1.0
        active_out_weights = active_out_weights / 2
        out_weights_ls = [active_out_weights for _ in range(sub_inter_dim // 4)]
        if sub_inter_dim % 4 != 0:
            inactive_out_weights = torch.ones(2)
            inactive_out_weights[-1] = -1.0
            out_weights_ls.append(inactive_out_weights)

        return torch.block_diag(*inter_weights_ls), torch.block_diag(*out_weights_ls)

    def _init_unary_operator_layer(
        self, subtree_ls: List[ParseTree]
    ) -> List[torch.nn.Module]:
        """
        Initializes the all the layers associated with carrying out the operators
        involved in the given list of subtrees.

        It assumes that the given list of subtrees contains only unary operators.

        Args:
            subtree_ls (list[ParseTree]): List of ParseTree instances with the same height
                in the full ParseTree that only has unary operators.

        Returns:
            list[torch.nn.Module]: List of neural network layers for unary operators.
        """
        n_nodes = len(subtree_ls)

        # assign weights
        weights = torch.zeros((n_nodes, n_nodes))
        for i, subtree in enumerate(subtree_ls):
            if subtree.node == "Not":
                weights[i, i] = -1.0
            elif subtree.node == "Id":
                weights[i, i] = 1.0

        # define layer
        layer = nn.Linear(n_nodes, n_nodes, bias=False)
        layer.weight = nn.Parameter(weights)
        return [layer]

    def _count_binary_operators(self, subtree_ls: List[ParseTree]) -> int:
        """
        Counts the occurrences of binary operators in the given list of ParseTree instances.

        Args:
            subtree_ls (list[ParseTree]): List of ParseTree instances with the same height
                in the full ParseTree.

        Returns:
            int: Count of binary operators.
        """
        return sum(
            [
                1
                for subtree in subtree_ls
                if subtree.node == "Or" or subtree.node == "And"
            ]
        )

    def _post_block_arg_num_ls(self, arg_num_ls: List[int]) -> List[int]:
        """
        Returns the list of number of arguments of each subtree after a binary operator block
        of layers.

        Args:
            arg_num_ls (list[int]): List of numbers of arguments for each subtree.

        Returns:
            list[int]: Number of arguments for each subtree after a block.
        """
        return [int((arg_num + arg_num % 2) / 2) for arg_num in arg_num_ls]


if __name__ == "__main__":
    from sympy import dotprint

    x, y, z, w = symbols("x y z w")
    k = x | (y & z & x) | ~w
    print(srepr(k))
    print(dotprint(k))
    tree = ParseTree(k)
