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
        
        result.append(var)
        seen.add(var.unique_id)

        if not var.is_leaf():
            for node in var.parents:
                dfs(node)
    dfs(variable)
    return result

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    nodes = list(topological_sort(variable))
    print(nodes)
    node_ids = {node.unique_id:node for node in nodes}
    print(node_ids)
    derivatives = {variable.unique_id: deriv}

    for node in nodes:
        if node.unique_id not in derivatives:
            continue
        
        # skip leaf for now
        if node.is_leaf():
            continue

        chain_deriv = node.chain_rule(derivatives[node.unique_id])

        for var, der in chain_deriv:
            if var.is_constant():
                continue
            var_id = var.unique_id
            if var_id not in derivatives:
                derivatives[var_id] = der
            else:
                derivatives[var_id] += der

    # Second pass: accumulate derivatives for all leaf nodes
    print(derivatives)
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
