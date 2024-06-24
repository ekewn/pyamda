from collections import deque
from functools import partial, reduce
from itertools import accumulate, count, filterfalse, islice, repeat, tee
import operator as op
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

#
#
# DATA
#
#


type IO                 = None
type FnN[a]             = Callable[[], a]           # Nullary Function
type FnU[a, b]          = Callable[[a], b]          # Unary...
type FnB[a, b, c]       = Callable[[a, b], c]       # Binary...
type FnT[a, b, c, d]    = Callable[[a, b, c], d]    # Ternary...
type FnQ[a, b, c, d, e] = Callable[[a, b, c, d], e] # Quaternary...
type FnUIO[a]           = Callable[[a], IO]
type Predicate[a]       = FnU[a, bool]


#
#
# FUNCTIONS
#
#

# Curried Classics

def map_[a, b](fn: Callable[[a], b]) -> Callable[[Iterable[a]], Iterable[b]]:
    """
    Curried map.
    """
    return partial(map, fn)


def filter_[a](p: Predicate[a]) -> Callable[[Iterable[a]], Iterable[a]]:
    """
    Curried filter.
    """
    return partial(filter, p)


def add[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.add. Returns unary function that adds this arg.
    """
    return partial(op.add, arg)


def sub[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts this arg.
    """
    return partial(op.sub, arg)


def mul[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.mul. Returns unary function that multiplies by this arg.
    """
    return partial(op.mul, arg)


def div_this[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the numerator as this arg.
    """
    return partial(op.floordiv, arg)


# Composers

def compose(*funcs: Callable) -> Callable:
    """
    Composes functions from left to right.
    """
    def compose2[a, b, c](x: Callable[[a], b], y: Callable[[b], c]) -> Callable[[a], c]:
        return lambda val: y(x(val))

    return reduce(compose2, funcs)


def pipe(val, *funcs: Callable):
    """
    Applies the functions to the value from left to right.
    """
    return compose(*funcs)(val)


# Composition Helpers

def id[a](x: a) -> a:
    """
    The identity property. Returns the argument.
    """
    return x


def always[a](x: a) -> FnN[a]:
    """
    Returns a function that always returns the arg.
    """
    return partial(id, x)


def tap[a](fn: Callable, x: a) -> a:
    """
    Calls a function and then returns the argument.
    """
    return compose(fn, id)(x)


# Logical

def T(*args) -> bool:
    """
    Always returns true.
    """
    return True


def F(*args) -> bool:
    """
    Always returns False.
    """
    return False


def both[a](p1: Predicate[a], p2: Predicate[a]) -> Predicate[a]:
    """
    Returns a function that returns True if both of the predicates are true.
    """
    def _(x, y, arg) -> bool: return x(arg) and y(arg)
    return partial(_, p1, p2)


def either[a](p1: Predicate[a], p2: Predicate[a]) -> Predicate[a]:
    """
    Returns a function that returns True if either of the predicates are true.
    """
    def _(x, y, arg) -> bool: return x(arg) or y(arg)
    return partial(_, p1, p2)


# Branches

def if_else[a, b, c](p: Predicate[a], if_true: FnU[a, b], if_false: FnU[a , c]) -> FnU[a, b | c]:
    """
    Functional ternary operator.
    """
    def _(p, t, f, v): return t(v) if p(v) else f(v)
    return partial(_, p, if_true, if_false)


def unless[a, b](p: Predicate[a], fn: FnU[a, b]) -> FnU[a, a | b]:
    """
    Returns a unary function that only applies the fn param if predicate is false, else returns the arg.
    """
    def _(p, f, v): return f(v) if not p(v) else v
    return partial(_, p, fn)
# NOTE: Needs test


def when[a, b](p: Predicate[a], fn: FnU[a, b]) -> FnU[a, a | b]:
    """
    Returns a unary function that only applies the fn param if predicate is true, else returns the arg.
    """
    def _(p, f, v): return f(v) if p(v) else v
    return partial(_, p, fn)
# NOTE: Needs test


type IfThens[a, b] = Tuple[Predicate[a], FnU[a, b]]
def cond[a, b](if_thens: List[IfThens[a, b]]) -> FnU[a, Optional[b]]:
    """
    Returns a unary function that applies the first function whose predicate is satisfied.
    """
    def _(its: List[IfThens[a, b]], arg: a):
        for it in its:
            if it[0](arg):
                return it[1](arg)
    return partial(_, if_thens)
# NOTE: Needs test


def const[a](x: a) -> Callable[[Any], a]:
    """
    Returns a unary function that always returns the argument to const, and ignores the arg to the resulting function.
    """
    def _(val, ignore): return val      # "Ignore is not accessed"... that's the point
    return partial(_, x)
# NOTE: Needs test


def default_to[a](default: a, val: a) -> a:
    """
    Returns default value if val is None.
    """
    return default if val is None else val
# NOTE: Needs test


def default_with[a, b](default: b, fn: FnU[a, Optional[b]]) -> FnU[a, b]:
    """
    Returns a function that will return the default value if the result is null.
    """
    def _(d, f, v): return d if f(v) is None else f(v)
    return partial(_, default, fn)
# NOTE: Needs test


# Container-related

def empty[a: (List, Dict, int, str)](x: a) -> a:
    """
    Returns the empty value (identity) of the monoid.
    e.g. [], {}, "", or 0.
    """
    is_a = partial(isinstance, x)
    if is_a(List):
        return [] #type:ignore
    elif is_a(Dict):
        return {} #type:ignore
    elif is_a(int):
        return 0 #type:ignore
    else:
        return "" #type:ignore
# NOTE: Needs test


def is_empty[a: (List, Dict, int, str)](x: a) -> bool:
    """
    Checks if value is the identity value of the monoid.
    """
    return any([x == [], x == {}, x == "", x == 0])
# NOTE: Needs test


def is_nil(x: Any) -> bool:
    """
    Checks if value is None.
    """
    return x is None
# NOTE: Needs test


# Iterator Specifics

def consume(i: Iterator) -> None:
    """
    Consumes an iterable to trigger side effects (avoids wasting the creation of a list).
    """
    deque(i, maxlen=0)


def take[a](n: int, i: Iterator[a]) -> Iterator[a]:
    """
    Returns an iterator of the first n items from the supplied iterator.
    """
    return islice(i, n)
# NOTE: Needs test


def head[a](i: Iterator[a]) -> a:
    """
    Gets first item from an iterator.
    """
    return next(i)
# NOTE: Needs test


def drop[a](n: int, i: Iterator[a]) -> Iterator[a]: 
    """
    Drops the first n items from an iterator.
    """
    return islice(i, n, None)
# NOTE: Needs test


def tail[a](i: Iterator[a]) -> Iterator[a]:
    """
    Returns an iterator without the first element of the given iterator.
    """
    return drop(1, i)
# NOTE: Needs test


def iterate[a](fn: Callable[[a], a], x: a) -> Iterator[a]:
    """
    Creates an iterator by applying the same function to the result of f(x).
    """
    return accumulate(repeat(x), lambda fx, _ : fn(fx))


def partition[a](p: Predicate[a], i: Iterable[a]) -> Tuple[Iterator[a], Iterator[a]]:
    """
    Returns the iterable separated into those that satisfy and don't satisfy the predicate.
    """
    t1, t2 = tee(i)
    return filter(p, t1), filterfalse(p, t2)
# NOTE: Needs test


# List Functions

def adjust[a](idx: int, fn:FnU[a, a], l: List[a]) -> List[a]:
    """
    Returns a copy of the given list with the element at the index transformed by the given function. The original list remains unchanged.
    """
    l2 = l.copy()
    l2[idx] = fn(l2[idx])
    return l2
# NOTE: Needs test


# Dictionary Functions

def get[a, b](d: Dict[a, b], default: b, key: a) -> b:
    """
    Dict.get alias.
    """
    return d.get(key, default)


#
#
# TESTS
#
#


if __name__ == "__main__":
    # Curried Classics
    assert list(take(3, map(partial(add,1), count()))) == list(take(3, map_(partial(add,1))(count())))
    assert list(take(3, filter(lambda x: x > 2, count()))) == list(take(3,filter_(lambda x: x > 2)(count())))
    # add, sub, mul, div_this

    # Composers
    assert pipe(1, partial(add, 1), partial(mul, 3)) == (1 + 1) * 3
    assert compose(len, lambda x: x + 10, lambda y: y - 1)("number should be 28") == 28

    # Composition Helpers
    assert id("test") == "test"
    assert always("test")() == "test"
    assert tap(partial(add, 1), 1) == 1

    # Logical
    assert T() == True
    assert F() == False
    # both
    # either


    assert list(take(4, iterate(partial(add, 3),2))) == [2, 5, 8, 11]
    assert list(take(3, drop(2, count()))) == [2, 3, 4]
    assert if_else(lambda _: True, lambda _: "a", lambda _: "b")("") == "a"
    assert if_else(lambda _: False, lambda _: "a", lambda _: "b")("") == "b"
    assert id("1") == "1"
    assert tap(id, "2") == "2"
    assert get({"a" : 1, "b" : 2}, "defaultvalue", "a") == 1
    assert get({"a" : 1, "b" : 2}, "defaultvalue", "c") == "defaultvalue"


