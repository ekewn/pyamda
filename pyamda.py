import operator as op
from collections import deque
from functools import partial, reduce
from itertools import accumulate, count, filterfalse, islice, repeat, tee
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple)

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

# Curried Built-ins

def map_[a, b](fn: FnU[a, b]) -> FnU[Iterable[a], Iterable[a]]:
    """
    Curried map.
    """
    return partial(map, fn)


def filter_[a](p: Predicate[a]) -> FnU[Iterable[a], Iterable[a]]:
    """
    Curried filter.
    """
    return partial(filter, p)


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


def flip[a, b, c](fn: FnB[a, b, c]) -> FnB[b, a, c]:
    """
    Returns a binary function with the argument order flipped.
    """
    def _(x: b, y: a): return fn(y, x)
    return _


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


def eq(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.eq.
    e.g. ( eq(1)(2) ) == ( 1 == 2 )
    """
    return partial(op.eq, x)


def gt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.gt.
    e.g. ( gt(1)(2) ) == ( 2 > 1 )
    """
    return partial(flip(op.gt), x)


def ge(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.ge.
    e.g. ( ge(1)(2) ) == ( 2 >= 1 )
    """
    return partial(flip(op.ge), x)


def lt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.lt.
    e.g. ( lt(1)(2) ) == ( 2 < 1 )
    """
    return partial(flip(op.lt), x)


def le(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.le.
    e.g. ( le(1)(2) ) == ( 2 <= 1 )
    """
    return partial(flip(op.le), x)


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


def when[a, b](p: Predicate[a], fn: FnU[a, b]) -> FnU[a, a | b]:
    """
    Returns a unary function that only applies the fn param if predicate is true, else returns the arg.
    """
    def _(p, f, v): return f(v) if p(v) else v
    return partial(_, p, fn)


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


def const[a](x: a) -> Callable[[Any], a]:
    """
    Returns a unary function that always returns the argument to const, and ignores the arg to the resulting function.
    """
    def _(val, ignore): return val      # "Ignore is not accessed"... that's the point
    return partial(_, x)


def none(*args) -> None:
    """
    Returns a function that always returns None.
    """
    return None


def default_to[a](default: a, val: a) -> a:
    """
    Returns default value if val is None.
    """
    return default if val is None else val


def default_with[a, b](default: b, fn: FnU[a, Optional[b]]) -> FnU[a, b]:
    """
    Returns a function that will return the default value if the result is null.
    """
    def _(d, f, v): return d if f(v) is None else f(v)
    return partial(_, default, fn)


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


def is_empty[a: (List, Dict, int, str)](x: a) -> bool:
    """
    Checks if value is the identity value of the monoid.
    """
    return any([x == [], x == {}, x == "", x == 0])


def is_none(x: Any) -> bool:
    """
    Checks if value is None.
    """
    return x is None


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


def head[a](i: Iterator[a]) -> a:
    """
    Gets first item from an iterator.
    """
    return next(i)


def drop[a](n: int, i: Iterator[a]) -> Iterator[a]: 
    """
    Drops the first n items from an iterator.
    """
    return islice(i, n, None)


def tail[a](i: Iterator[a]) -> Iterator[a]:
    """
    Returns an iterator without the first element of the given iterator.
    """
    return drop(1, i)


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


# List Functions

def adjust[a](idx: int, fn:FnU[a, a], l: List[a]) -> List[a]:
    """
    Returns a copy of the given list with the element at the index transformed by the given function. The original list remains unchanged.
    """
    l2 = l.copy()
    l2[idx] = fn(l2[idx])
    return l2


# Dictionary Functions

def get[a, b](d: Dict[a, b], default: b, key: a) -> b:
    """
    Dict.get alias.
    """
    return d.get(key, default)


# Mathematical Functions
def add_this[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.add. Returns unary function that adds this arg.
    """
    return partial(op.add, arg)


def sub_from[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts from this arg.
    """
    return partial(op.sub, arg)


def sub_this[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts this arg.
    """
    return partial(flip(op.sub), arg)


def mul_by[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.mul. Returns unary function that multiplies by this arg.
    """
    return partial(op.mul, arg)


def div_this[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the numerator as this arg.
    """
    return partial(op.floordiv, arg)


def div_by[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the denominator as this arg.
    """
    return partial(flip(op.floordiv), arg)




#
#
# TESTS
#
#


if __name__ == "__main__":
    # Curried Built-ins
    # assert list(take(3, map(add_this(1), count()))) == list(take(3, map_(add_this(1))(count())))
    # assert list(take(3, filter(gt(2), count())))    == list(take(3,filter_(gt(2))(count())))

    # Composers
    assert compose(len, add_this(10), sub_this(1))("number should be 28") == len("number should be 28") + 10 - 1
    assert pipe(1, add_this(1), mul_by(3))                                == (1 + 1) * 3

    # Composition Helpers
    assert id("test")          == "test"
    assert tap(id, "2")        == "2"
    assert always("test")()    == "test"
    assert tap(add_this(1), 1) == 2

    # Logical
    assert T()
    assert not F()
    assert both      (T, T)("anything")
    assert not both  (T, F)("anything")
    assert not both  (F, F)("anything")
    assert either    (T, T)("anything")
    assert either    (T, F)("anything")
    assert not either(F, F)("anything")
    assert not eq(1)(0)
    assert eq    (1)(1)
    assert not eq(1)(2)
    assert not gt(1)(0)
    assert not gt(1)(1)
    assert gt    (1)(2)
    assert not ge(1)(0)
    assert ge    (1)(1)
    assert ge    (1)(2)
    assert lt    (1)(0)
    assert not lt(1)(1)
    assert not lt(1)(2)
    assert le    (1)(0)
    assert le    (1)(1)
    assert not le(1)(2)
    
    # Branches
    assert if_else     (T, const      ("a"), const("b"))("anything") == "a"
    assert if_else     (F, const      ("a"), const("b"))("anything") == "b"
    assert unless      (T, add_this   (1))        (1)                == 1
    assert unless      (F, add_this   (1))        (1)                == 2
    assert when        (T, add_this   (1))        (1)                == 2
    assert when        (F, add_this   (1))        (1)                == 1
    assert const       (1)            ("anything")                   == 1
    assert const       ("this")       (1)                            == "this"
    assert default_to  (10, 11)                                      == 11
    assert default_to  (10, None)                                    == 10
    assert default_with(10, add_this  (1))        (20)               == 21
    assert default_with(10, none)     (20)                           == 10
    condtest: FnU[int, str] = default_with("otherwise", cond([(gt(0), const("is positive"))
                                                               , (eq(0), const("is zero"))
                                                               , (lt(0), const("is negative"))]))
    assert condtest(1) == "is positive"
    assert condtest(0) == "is zero"
    assert condtest(-1) == "is negative"

    # Container-related
    assert all([is_empty(empty(["list"]))
               , is_empty(empty({"dict" : 1}))
               , is_empty(empty(123))
               , is_empty(empty("string"))])
    assert not all([is_empty(["this should fail because I'm not empty"])
               , is_empty({})
               , is_empty(0)
               , is_empty("")])
    assert is_none(None)
    assert not is_none("this should fail because I'm not None")

    # Iterator Specifics
    assert list(take(4, iterate(add_this(3), 2))) == [2, 5, 8, 11]
    assert list(take(3, drop(2, count())))        == [2, 3, 4]
    assert head(count())                          == 0
    assert list(take(3, tail(count())))           == [1, 2, 3]
    #partition

    # List Functions
    #adjust

    # Math Functions
    assert add_this(1)(7) == 1 + 7
    assert mul_by(3)(7)   == 3 * 7
    assert sub_from(7)(3) == 7 - 3
    assert sub_this(3)(7) == 7 - 3
    assert div_this(8)(4) == 8 / 4
    assert div_by(4)(8)   == 8 / 4

