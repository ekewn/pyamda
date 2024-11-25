import operator as op
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Iterable,
    NamedTuple,
    Optional,
    Dict,
    List,
    Container,
    Tuple,
)
from functools import reduce

type IO = None
type FnN[a] = Callable[[], a]  # Nullary Function i.e. takes no arguments
type FnU[a, b] = Callable[[a], b]  # Unary i.e. takes one argument
type FnB[a, b, c] = Callable[[a, b], c]  # Binary...
type FnT[a, b, c, d] = Callable[[a, b, c], d]  # Ternary...
type FnQ[a, b, c, d, e] = Callable[[a, b, c, d], e]  # Quaternary...
type FnUIO[a] = Callable[[a], IO]
type Predicate[a] = FnU[a, bool]


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

    >>> i = [0, 1, 2]
    >>> fn = lambda x: x + 1
    >>> assert list(map_(fn)(i)) == list(map(fn, i)) == [1, 2, 3]
    """
    return partial(map, fn)


def filter_[a](p: Predicate[a]) -> FnU[Iterable[a], Iterable[a]]:
    """
    Curried filter.

    >>> i = [0, 1, 2]
    >>> p = lambda x: x == 1
    >>> assert list(filter_(p)(i)) == list(filter(p, i)) == [0, 2]
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

    def _(x: b, y: a):
        return fn(y, x)

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

    def _(val, *args):
        return val

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

    def _(d, f, v):
        return d if f(v) is None else f(v)

    return partial(_, default, fn)


# Conditionals


def if_else[
    a, b, c
](p: Predicate[a], if_true: FnU[a, b], if_false: FnU[a, c]) -> FnU[a, b | c]:
    """
    Functional ternary operator.
    """

    def _(p, t, f, v):
        return t(v) if p(v) else f(v)

    return partial(_, p, if_true, if_false)


def unless[a, b](p: Predicate[a], fn: FnU[a, b]) -> FnU[a, a | b]:
    """
    Returns a unary function that only applies the fn param if predicate is false, else returns the arg.
    """
    return if_else(p, identity, fn)


def when[a, b](p: Predicate[a], fn: FnU[a, b]) -> FnU[a, a | b]:
    """
    Returns a unary function that only applies the fn param if predicate is true, else returns the arg.
    """
    return if_else(p, fn, identity)


def optionally[a, b](fn: FnU[a, b]) -> FnU[Optional[a], Optional[b]]:
    """
    Abstracts the common flow of only working on non-none values in if_else blocks.
    Function will only call if the value is not none, else none will be passed along.
    """

    def _(fn: FnU[a, b], v: Optional[a]):
        if v is not None:
            return fn(v)
        else:
            return v

    return partial(_, fn)


def cond[a, b](if_thens: List[Tuple[Predicate[a], FnU[a, b]]]) -> FnU[a, Optional[b]]:
    """
    Returns a unary function that applies the first function whose predicate is satisfied.
    """

    def _(its: List[Tuple[Predicate[a], FnU[a, b]]], arg: a):
        for it in its:
            if it[0](arg):
                return it[1](arg)

    return partial(_, if_thens)


def try_except[
    a, b
](tryer: FnU[a, b], excepter: FnB[a, Exception, Exception]) -> FnU[a, b | Exception]:
    """
    Guards a formula that might throw an error. Will catch and run the provided excepter formula.
    """

    def _(t, e, v):
        try:
            return t(v)
        except Exception as err:
            return e(v, err)

    return partial(_, tryer, excepter)


def try_[a, b](tryer: FnU[a, b]) -> FnU[a, b | Exception]:
    """
    Guards a formula that might throw an error. If an exception is encountered, the exception will be returned
    with the arg nested in the exception i.e. you can retrieve it by doing err_val(Exception).
    """

    def _(t, v):
        try:
            return t(v)
        except Exception as err:
            return Exception(v, err)

    return partial(_, tryer)


def raise_(e: Exception) -> Exception:
    """
    A function that raises exceptions.
    """
    raise e


def optional[a, b](tryer: FnU[a, b]) -> FnU[a, Optional[b]]:
    """
    Guards a formula that might throw an error. Will return the None an exception occurs.
    """

    def _(t, v):
        try:
            return t(v)
        except:
            return None

    return partial(_, tryer)


def on_success[a, b](fn: FnU[a, b]) -> FnU[a | Exception, b | Exception]:
    """
    Abstracts the common flow of only working on non-err values in if_else blocks.
    Function will only call if the value is not an error, else the error will be passed along.
    """

    def _(fn: FnU[a, b], v: a | Exception):
        if not isinstance(v, Exception):
            return fn(v)
        else:
            return v

    return partial(_, fn)


def on_err[a, b](fn: FnU[Exception, b]) -> FnU[Exception | a, b | a]:
    """
    Abstracts the common flow of only working on err values in if_else blocks.
    Function will only call if the value is an error, else the value will be passed along.
    """

    def _(fn: FnU[Exception, b], v: Exception | a):
        if isinstance(v, Exception):
            return fn(v)
        else:
            return v

    return partial(_, fn)


def test[a, b](fn: FnU[a, b], cases: Iterable[Case[a, b]]) -> None:
    """
    Similar to apply_spec, where we apply the function to each "input" of the case and map the value to the output.
    Assert every application of the funtion to an input results in the expected output.
    """
    for case in cases:
        assert case.expected(
            fn(case.input)
        ), f"{fn.__name__} - {case.name} - Expected {case.expected} but got {fn(case.input)}."


# Container-related


def is_a[a](x: type) -> Predicate[a]:
    """
    Wrapper for isinstance check. Returns a predicate.
    """
    return partial(flip(isinstance), x)


def is_empty[a: (List, Dict, int, str)](x: a) -> bool:
    """
    Checks if value is the identity value of the monoid.
    """
    return any([x == [], x == {}, x == "", x == 0])


is_none: Predicate[Any] = lambda x: x is None
is_err: Predicate[Any] = is_a(Exception)
is_str: Predicate[Any] = is_a(str)
is_int: Predicate[Any] = is_a(int)
is_bool: Predicate[Any] = is_a(bool)
is_dict: Predicate[Any] = is_a(dict)
is_list: Predicate[Any] = is_a(list)
is_float: Predicate[Any] = is_a(float)


def is_namedtuple(x: object) -> bool:
    """
    Not allowed to do isinstance checks on namedtuple, so this will generally
    provide the correct answer. It is possible to get a falsed positive if someone
    is really trying, but it will work in most conditions.
    """
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def empty[a: (List, Dict, int, str)](x: a) -> a:
    """
    Returns the empty value (identity) of the monoid.
    e.g. [], {}, "", or 0.
    """
    if is_list(x):
        assert isinstance(x, List)
        return []
    elif is_dict(x):
        assert isinstance(x, Dict)
        return {}
    elif is_int(x):
        assert isinstance(x, int)
        return 0
    else:
        assert isinstance(x, str)
        return ""


def err_val(x: Exception, idx: int = 0) -> Any:
    """
    Gets the value/text out of an exception.
    Note: Exceptions aren't parameterize-able so we can't determine the type of value we get out of it.
    """
    return x.args[idx]


def contains(x: Container[object]) -> Predicate[object]:
    """
    Curried version of operator.contains
    e.g. ( contains([0, 1, 2])(1) == ( 1 in [0, 1, 2] ) == True
    """
    return partial(op.contains, x)


def in_(x: object) -> Predicate[object]:
    """
    Curried version of "is in"
    e.g. ( in_(1)([0, 1, 2]) == ( 1 in [0, 1, 2] ) == True
    """

    def _(x, y):
        return x in y

    return partial(_, x)
