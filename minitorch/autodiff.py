from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    vals_list = list(vals)
    
    vals_plus = vals_list.copy()
    vals_plus[arg] += epsilon
    
    vals_minus = vals_list.copy()
    vals_minus[arg] -= epsilon
    
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    
    seen = set()
    result = []

    def dfs(var: Variable) -> None:
        if var.is_constant():
            return
        
        if var.unique_id in seen:
            return
        
        seen.add(var.unique_id)

        if not var.is_leaf():
            for node in var.parents:
                dfs(node)
        
        result.append(var)
    dfs(variable)
    return reversed(result)

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # 1. Call topological sort to get an ordered queue
    nodes = list(topological_sort(variable))
    
    # 2. Create a dictionary of Scalars and current derivatives
    derivatives = {variable.unique_id: deriv}

    # 3. For each node in backward order
    for node in nodes:
        # Pull a completed Scalar and derivative from the queue
        if node.unique_id not in derivatives:
            continue
        
        d_out = derivatives[node.unique_id]
        
        # a. If the Scalar is a leaf, we'll accumulate its derivative later
        # (after all paths have been processed)
        if node.is_leaf():
            continue
        
        # b. If the Scalar is not a leaf:
        # i. Call .chain_rule on the last function with dout
        chain_deriv = node.chain_rule(d_out)
        
        # ii. Loop through all the Scalars+derivative produced by the chain rule
        for var, der in chain_deriv:
            # Skip constants
            if var.is_constant():
                continue
            
            # iii. Accumulate derivatives for the Scalar in a dictionary
            var_id = var.unique_id
            if var_id not in derivatives:
                derivatives[var_id] = der
            else:
                derivatives[var_id] += der
    
    # 4. After all derivatives are accumulated, update leaf nodes
    for node in nodes:
        if node.is_leaf() and node.unique_id in derivatives:
            node.accumulate_derivative(derivatives[node.unique_id])

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
