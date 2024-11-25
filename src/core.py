import operator as op
from functools import partial, reduce
from typing import (Any, Callable, Iterable, NamedTuple,
                    Optional)
from functools import reduce

type IO                 = None
type FnN[a]             = Callable[[], a]           # Nullary Function i.e. takes no arguments
type FnU[a, b]          = Callable[[a], b]          # Unary i.e. takes one argument
type FnB[a, b, c]       = Callable[[a, b], c]       # Binary...
type FnT[a, b, c, d]    = Callable[[a, b, c], d]    # Ternary...
type FnQ[a, b, c, d, e] = Callable[[a, b, c, d], e] # Quaternary...
type FnUIO[a]           = Callable[[a], IO]
type Predicate[a]       = FnU[a, bool]

class Case[a, b](NamedTuple):
    name: str
    input: a
    expected: Predicate[b]


#
#
# FUNCTIONS
#
#

# Aliases

item = op.itemgetter
method = op.methodcaller
p = partial


# Curried Built-ins

def map_[a, b](fn: FnU[a, b]) -> FnU[Iterable[a], Iterable[b]]:
    """
    Curried map.
    e.g map_(fn)(iter) == map(fn, iter)
    """
    return partial(map, fn)


def filter_[a](p: Predicate[a]) -> FnU[Iterable[a], Iterable[a]]:
    """
    Curried filter.
    e.g. filter_(predicate)(iter) == filter(predicate, iter)
    """
    return partial(filter, p)


def print_[a](msg: str) -> FnU[a, a]:
    """
    Like, tap(print)(), but can print an arbitrary message. 
    Used for printing within a pipe expression. Print the message, and return whatever 
    value is passed to it.
    e.g. print_("my message)(1) == print("my message")(1) == 1
    """
    def _(msg, x) -> a:
        print(msg)
        return x
    return partial(_, msg)


def assert_[a](p: Predicate[a]) -> FnU[a, a]:
    """
    Funcational assert statement. Asserts the predicate holds with the value, then returns the value.
    """
    def _(p, x: a) -> a:
        assert p(x), f"Asserton failed with predicate {p} and value {x}"
        return x
    return partial(_, p)


# Composition Pipeline Essentials

def compose(*funcs: Callable) -> Callable:
    """
    Composes functions from left to right.
    e.g compose(add_to(1), mul_by(2))(3) == (3 + 1) * 2 == 8
    """
    def compose2[a, b, c](x: Callable[[a], b], y: Callable[[b], c]) -> Callable[[a], c]:
        return lambda val: y(x(val))
    return reduce(compose2, funcs)


def pipe(val, *funcs: Callable) -> Any:
    """
    Applies the functions to the value from left to right.
    e.g. pipe(3, add_to(1), mul_by(2)) == (3 + 1) * 2 == 8
    """
    return compose(*funcs)(val)


def foreach[a](fn: FnU[a, None]) -> FnU[Iterable[a], Iterable[a]]:
    """
    Like map but returns the original array. Used for performing side effects.
    The benefit of returning the original array is that you can reuse your final data
    to do mulitple side effects.
    """
    def _(fn, i) -> Iterable[a]:
        for x in i:
            fn(x)
        return i
    return partial(_, fn)


def identity[a](x: a) -> a:
    """
    The identity property. Returns the argument.
    e.g. id(1) == 1
    """
    return x


def always[a](x: a) -> FnN[a]:
    """
    Returns a function that always returns the arg.
    e.g. always(10)() == 10
    """
    return partial(identity, x)


def flip[a, b, c](fn: FnB[a, b, c]) -> FnB[b, a, c]:
    """
    Returns a binary function with the argument order flipped.
    e.g. flip(a -> b -> c) == b -> a -> c
    """
    def _(x: b, y: a): return fn(y, x)
    return _


def tap[a](fn: Callable, x: a) -> a:
    """
    Calls a function and then returns the argument.
    e.g. tap(compose(print, add_to(1), print), 2) == print(2), add 1, print(3), return 2
    """
    fn(x)
    return x


def print_arg[a](x: a) -> a:
    """
    Prints the argument given to it, then returns the value.
    Same as partial(tap, print)(x).
    """
    print(x)
    return x


def const[a](x: a) -> FnU[Any, a]:
    """
    Returns a unary function that always returns the argument to const, and ignores the arg to the resulting function.
    e.g. c = const(1)
    c("literally any arg") == 1
    """
    def _(val, *args): return val
    return partial(_, x)


def none(*args) -> None:
    """
    A function that always returns None.
    e.g. none() == None
    """
    return None


def default_to[a](default: a, val: a) -> a:
    """
    Returns default value if val is None.
    e.g.
    default_to("a", None)(None) == "a"
    default_to("a", None)("literally any arg") == "literally any arg"

    """
    return default if val is None else val


def default_with[a, b](default: b, fn: FnU[a, Optional[b]]) -> FnU[a, b]:
    """
    Returns a function that replaces the None case of the given function with the default value.
    e.g.
    func_with_defaulting = default_with("my default value", func_that_could_return_none)
    """
    def _(d, f, v): return d if f(v) is None else f(v)
    return partial(_, default, fn)


