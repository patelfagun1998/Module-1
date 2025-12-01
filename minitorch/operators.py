"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:

    return x*y

def id(x: float) -> float:

    return x

def add(x: float, y: float) -> float:

    return x+y

def neg(x: float) -> float:

    return float(-x)

def lt(first: float, second: float) -> bool:

    return first < second

def eq(x: float, y: float) -> float:

    return x == y

def max(x: float, y: float) -> float:

    return x if x > y else y

def is_close(x: float, y: float) -> float:

    return abs(x-y) < math.e**(-2)

def sigmoid(x: float) -> float:

    return 1.0/(1.0+math.e**(-x)) if x >= 0 else math.e**x/(1.0+math.e**x)

def relu(x: float) -> float:

    return x if x >= 0 else 0.0

def log(x: float) -> float:

    return math.log(x)

def exp(x: float) -> float:

    return math.e**x

def inv(x: float) -> float:

    return 1.0/x

def log_back(x: float, y: float) -> float:

    return y/x

def inv_back(x: float, y: float) -> float:

    return -y/x**2

def relu_back(x: float, y: float) -> float:

    return 0 if x <= 0 else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

from typing import Callable, List, Any, Iterable

def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:

    return [fn(x) for x in lst]

def zipWith(fn: Callable[[float, float], float], lst1: Iterable[float], lst2:Iterable[float]) -> Iterable[float]:

    return [fn(a,b) for a, b in zip(lst1, lst2)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float]) -> float:

    print(lst)
    if not lst:
        return 0
    it = iter(lst)
    result = next(it)
    
    for itm in it:

        result = fn(result, itm)
        
    return result

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def negate(x: float) -> float:

    return -x

def negList(lst: List[float]):

    return map(negate, lst)

def addLists(lst1: List[float], lst2: List[float]) -> Iterable[float]:

    return zipWith(add, lst1, lst2)

def sum(lst1: List[float]) -> float:

    return reduce(add, lst1)

def prod(lst1: List[float]) -> float:

    return reduce(mul, lst1)

# TODO: Implement for Task 0.3.
