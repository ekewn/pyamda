import operator as op
from collections import deque
from functools import partial, reduce
from itertools import (accumulate, chain, count, filterfalse, islice, repeat,
                       tee)
from typing import (Any, Callable, Container, Dict, Iterable, Iterator, List,
                    NoReturn, Optional, Tuple)

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


# Composers

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


# Composition Helpers

def id[a](x: a) -> a:
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
    return partial(id, x)


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
    return compose(fn, id)(x)


# Logical

def T(*args) -> bool:
    """
    Always returns True.
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
    e.g. ( eq(1)(2) ) == ( 1 == 2 ) == False
    """
    return partial(op.eq, x)


def gt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.gt.
    e.g. ( gt(1)(2) ) == ( 2 > 1 ) == True
    """
    return partial(flip(op.gt), x)


def ge(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.ge.
    e.g. ( ge(1)(2) ) == ( 2 >= 1 ) == True
    """
    return partial(flip(op.ge), x)


def lt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.lt.
    e.g. ( lt(1)(2) ) == ( 2 < 1 ) == False
    """
    return partial(flip(op.lt), x)


def le(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.le.
    e.g. ( le(1)(2) ) == ( 2 <= 1 ) == False
    """
    return partial(flip(op.le), x)


def is_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.is_.
    e.g. ( is_(1)(2) ) == ( 1 is 2 ) == False
    """
    return partial(op.is_, x)


def is_not(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.is_not.
    e.g. ( is_not(1)(2) ) == ( 1 is not 2 ) == True
    """
    return partial(op.is_not, x)


def and_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.and_.
    e.g. ( and_(True)(False) ) == ( True and False ) == False
    """
    return partial(op.and_, x)


def or_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.or_.
    e.g. ( or_(True)(False) ) == ( True or False ) == True
    """
    return partial(op.or_, x)


# Control Flow

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
    return if_else(p, id, fn)


def when[a, b](p: Predicate[a], fn: FnU[a, b]) -> FnU[a, a | b]:
    """
    Returns a unary function that only applies the fn param if predicate is true, else returns the arg.
    """
    return if_else(p, fn, id)


def cond[a, b](if_thens: List[Tuple[Predicate[a], FnU[a, b]]]) -> FnU[a, Optional[b]]:
    """
    Returns a unary function that applies the first function whose predicate is satisfied.
    """
    def _(its: List[Tuple[Predicate[a], FnU[a, b]]], arg: a):
        for it in its:
            if it[0](arg):
                return it[1](arg)
    return partial(_, if_thens)


def const[a](x: a) -> FnU[Any, a]:
    """
    Returns a unary function that always returns the argument to const, and ignores the arg to the resulting function.
    """
    def _(val, ignore_any_other_args): return val
    return partial(_, x)


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


def try_except[a, b](tryer: FnU[a, b], excepter: FnB[a, Exception, Exception]) -> FnU[a, b | Exception]:
    """
    Guards a formula that might throw an error. Will catch and run the provided excepter formula.
    """
    def _(t, e, v):
        try: return t(v)
        except Exception as err: return e(v, err)
    return partial(_, tryer, excepter)


def try_[a, b](tryer: FnU[a, b]) -> FnU[a, b | Exception]:
    """
    Guards a formula that might throw an error. If an exception is encountered, the exception will be returned
    with the arg prepended to the exception body.
    """
    def _(t, v):
        try: return t(v)
        except Exception as err: return Exception(f"Arg: {v} \n Err: {err}")
    return partial(_, tryer)


def optional[a, b](tryer: FnU[a, b]) -> FnU[a, Optional[b]]:
    """
    Guards a formula that might throw an error. Will return the None an exception occurs.
    """
    def _(t, v):
        try: return t(v)
        except: return None
    return partial(_, tryer)


def on_success[a, b](fn: FnU[a, b]) -> FnU[a | Exception, b | Exception]:
    """
    Abstracts the common flow of only working on non-err values in if_else blocks.
    Function will only call if the value is not an error, else the error will be passed along.
    """
    def _(fn: FnU[a, b], v: a | Exception):
        if not isinstance(v, Exception): return fn(v)
        else: return v
    return partial(_, fn)


def on_err[a, b](fn: FnU[Exception, b]) -> FnU[Exception | a, b | a]:
    """
    Abstracts the common flow of only working on err values in if_else blocks.
    Function will only call if the value is an error, else the value will be passed along.
    """
    def _(fn: FnU[Exception, b], v: Exception | a):
        if isinstance(v, Exception): return fn(v)
        else: return v
    return partial(_, fn)


def none(*args) -> None:
    """
    A function that always returns None.
    """
    return None


def except_(e: Exception) -> Exception:
    """
    A function that raises exceptions.
    """
    raise e


# Container-related

def empty[a: (List, Dict, int, str)](x: a) -> a:
    """
    Returns the empty value (identity) of the monoid.
    e.g. [], {}, "", or 0.
    """
    is_a = partial(isinstance, x)
    if is_a(List):
        assert isinstance(x, List)
        return []
    elif is_a(Dict):
        assert isinstance(x, Dict)
        return {} 
    elif is_a(int):
        assert isinstance(x, int)
        return 0 
    else:
        assert isinstance(x, str)
        return "" 


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


def is_err(x: Any) -> bool:
    """
    Checks if value is an Exception.
    """
    return isinstance(x, Exception)


def contains(x: Container[object]) -> Predicate[object]:
    """
    Curried version of operator.contains
    e.g. ( contains([0, 1, 2])(1) == ( 1 in [0, 1, 2] ) == True
    """
    return partial(op.contains, x)


# Iterable Generics

def count_of(x: object) -> FnU[Iterable[object], int]:
    """
    Curried version of operator.countOf.
    e.g. count_of(1)([1, 2, 1, 3]) == operator.countOf([1, 2, 1, 3], 1) == 2
    """
    return partial(flip(op.countOf), x)


# Iterator Specifics

def consume(i: Iterator) -> None:
    """
    Consumes an iterable to trigger side effects (avoids wasting the creation of a list).
    """
    deque(i, maxlen=0)


def take[a](n: int, i: Iterable[a]) -> Iterator[a]:
    """
    Returns an iterator of the first n items from the supplied iterable.
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

type NewList[a] = List[a]

def adjust[a](idx: int, fn:FnU[a, a], l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list with the element at the index transformed by the given function.
    """
    l2 = l.copy()
    l2[idx] = fn(l2[idx])
    return l2


def move[a](idx_from: int, idx_to: int, l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list with the the item at the specified index moved to the new index.
    """
    l2 = l.copy()
    l2.insert(idx_to, l2.pop(idx_from))
    return l2


def swap[a](idx1: int, idx2: int, l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list Swaps the items at the specified indices.
    """
    l2 = l.copy()
    v1, v2 = l[idx1], l[idx2]
    l2[idx1], l2[idx2] = v2, v1
    return l2


def update[a](idx: int, val: a, l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list with the item at the specified index updated.
    """
    l2 = l.copy()
    l2[idx] = val
    return l2


def cons[a](val: a | List[a], l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list with the value/other list prepended.
    """
    l2 =l.copy()
    if isinstance(val, List):
        assert isinstance(val, List)
        l2 = val + l2
    else:
        l2.insert(0, val)
    return l2


def remove[a](first: int, last: int, l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list with all items from first to last indices given (not including the value at the last index) removed.
    """
    l2 = l.copy()
    del l2[first:last]
    return l2


def startswith[a](val: a, l: List[a]) -> bool:
    """
    Does the list start with the given value?
    """
    return l[0] == val


def endswith[a](val: a, l: List[a]) -> bool:
    """
    Does the list end with the given value?
    """
    return l[len(l)-1] == val


# Dictionary Functions

def get[a, b](d: Dict[a, b], default: b, key: a) -> b:
    """
    Dict.get alias.
    """
    return d.get(key, default)


def prop[a, b](key: a) -> FnU[Dict[a, b], Optional[b]]:
    """
    Returns a function that given a dictionary returns the value at the key, if present, else None.
    """
    def _(k, d): return d.get(k)
    return partial(_, key)


def prop_or[a, b](key: a, default: b) -> FnU[Dict[a, b], b]:
    """
    Returns a function that given a dictionary returns the value at the key, if present, else the default.
    """
    def _(k, default, d): return d.get(k, default)
    return partial(_, key, default)


def prop_eq[a, b](key: a, val: b) -> FnU[Dict[a, b], bool]:
    """
    Returns a function that given a dictionary returns if the value at the key is equal to the val.
    """
    def _(k, val, d): return d.get(k) == val
    return partial(_, key, val)


def attr(name: str) -> FnU[object, Optional[Any]]:
    """
    Returns a function that given an object returns the value of the object attribute if present, or none.
    """
    def _(n, obj): 
        try:
            return op.attrgetter(n)(obj)
        except:
            return None
    return partial(_, name)


def attr_or(name: str, default: Any) -> FnU[object, Any]:
    """
    Returns a function that given an object returns the value of the object attribute if present, or none.
    """
    def _(n, obj): 
        try:
            return op.attrgetter(n)(obj)
        except:
            return default
    return partial(_, name)


def attr_eq(name: str, val: Any) -> FnU[object, bool]:
    """
    Returns a function that given an object returns if the value of the object attribute is equal to the val.
    """
    def _(n, v, o): 
        try:
            return op.attrgetter(n)(o) == v
        except:
            return False
    return partial(_, name, val)


# String Functions

type NewStr = str

def replace(old: str, new: str, s: str) -> NewStr:
    """
    Pure s.replace()
    """
    s2 = s
    return s2.replace(old, new)


def split(sep: str, s: str, maxsplits: int = -1) -> List[NewStr]:
    """
    Pure s.split().
    """
    s2 = s
    return s2.split(sep, maxsplits)


# Mathematical Functions

def add[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.add. Returns unary function that adds this arg.
    e.g. add(1)(1) == operator.add(1, 1) == 2
    """
    return partial(op.add, arg)


def sub_from[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts from this arg.
    e.g. sub_from(2)(1) == 2 - 1 == 1
    """
    return partial(op.sub, arg)


def sub_this[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts this arg.
    e.g. sub_this(2)(1) == 1 - 2 == (-1)
    """
    return partial(flip(op.sub), arg)


def mul[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.mul. Returns unary function that multiplies by this arg.
    e.g. mul(2)(3) == 2 * 3 == 6
    """
    return partial(op.mul, arg)


def div_this[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the numerator as this arg.
    e.g. div_this(6)(3) == 6 // 3 == 2
    """
    return partial(op.floordiv, arg)


def div_by[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the denominator as this arg.
    e.g. div_by(6)(3) == 3 // 6 == 0
    """
    return partial(flip(op.floordiv), arg)


#
#
# TESTS
#
#


if __name__ == "__main__":
    # Curried Built-ins
    assert list(take(3, map(add(1), count()))) == list(take(3, map_(add(1))(count())))
    assert list(take(3, filter(gt(2), count())))    == list(take(3,filter_(gt(2))(count())))

    # Composers
    assert compose(len, add(10), sub_this(1))("number should be 28") == len("number should be 28") + 10 - 1
    assert pipe(1, add(1), mul(3))                                == (1 + 1) * 3

    # Composition Helpers
    assert id("test")          == "test"
    assert tap(id, "2")        == "2"
    assert always("test")()    == "test"
    assert tap(add(1), 1) == 2

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
    assert is_(1)(1)
    assert not is_(1)(2)
    assert not is_not(1)(1)
    assert is_not(1)(2)
    assert and_(True)(True)
    assert not and_(True)(False)
    assert or_(True)(False)
    assert not or_(False)(False)
    assert contains([0, 1, 2])(1)
    assert not contains([0, 1, 2])(3)

    # Control Flow
    assert if_else     (T, const      ("a"), const("b"))("anything") == "a"
    assert if_else     (F, const      ("a"), const("b"))("anything") == "b"
    assert unless      (T, add   (1))        (1)                     == 1
    assert unless      (F, add   (1))        (1)                     == 2
    assert when        (T, add   (1))        (1)                     == 2
    assert when        (F, add   (1))        (1)                     == 1
    assert const       (1)            ("anything")                   == 1
    assert const       ("this")       (1)                            == "this"
    assert default_to  (10, 11)                                      == 11
    assert default_to  (10, None)                                    == 10
    assert default_with(10, add  (1))        (20)                    == 21
    assert default_with(10, none)     (20)                           == 10
    condtest: FnU[int, str]                                          =  default_with("otherwise", cond([(gt(0), const("is positive"))
                                                               , (eq(0), const("is zero"))
                                                               , (lt(0), const("is negative"))]))
    assert condtest(1) == "is positive"
    assert condtest(0) == "is zero"
    assert condtest(-1) == "is negative"
    assert isinstance(try_except(div_by(0), lambda v, e: Exception(f"arg: {v} caused err {e}"))(1), Exception)
    assert isinstance(try_(div_by(0))(1), Exception)
    assert optional(div_by(0))(1) is None
    assert optional(div_by(1))(1) == 1
    #on_success
    #on_err

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
    assert is_err(Exception("this is err!"))
    assert not is_err(1)

    # Iterable Generics
    assert count_of(1)([1, 1, 0, 1]) == op.countOf([1, 1, 0, 1], 1)

    # Iterator Specifics
    assert list(take(4, iterate(add(3), 2))) == [2, 5, 8, 11]
    assert list(take(3, drop(2, count())))        == [2, 3, 4]
    assert head(count())                          == 0
    assert list(take(3, tail(count())))           == [1, 2, 3]
    partitiontest1, partitiontest2                =  partition(gt(3), take(6, count()))
    assert list(partitiontest1)                   == [4, 5]
    assert list(partitiontest2)                   == [0, 1, 2, 3]

    # List Functions
    assert adjust(2, add(3), [0, 1, 2, 3]) == [0, 1, 5, 3]
    assert move(0, 2, [0, 1, 2, 3, 4]) == [1, 2, 0, 3, 4]
    assert swap(0, 2, [0, 1, 2, 3, 4]) == [2, 1, 0, 3, 4]
    assert update(0, 2, [0, 1, 2, 3, 4]) == [2, 1, 2, 3, 4]
    assert cons(0, [0, 1, 2, 3, 4]) == [0, 0, 1, 2, 3, 4]
    assert cons([0, 1], [0, 1, 2, 3, 4]) == [0, 1, 0, 1, 2, 3, 4]
    assert startswith(0, [0, 1, 2])
    assert not startswith(0, [1, 2])
    assert endswith(2, [0, 1, 2])
    assert not endswith(2, [0, 1])
    assert remove(0, 2, [0, 1, 2, 3]) == [2, 3]
    assert not remove(0, 2, [0, 1, 2, 3]) == [1, 2, 3]

    # Dictionary Functions
    dtest: Dict[str, str] = {"a": "1", "b": "2"}
    assert get(dtest, "default", "a") == "1"
    assert get(dtest, "default", "c") == "default"
    assert prop("a")(dtest) == "1"
    assert prop("c")(dtest) == None
    assert prop_or("a", "default")(dtest) == "1"
    assert prop_or("c", "default")(dtest) == "default"
    assert prop_eq("a", "1")(dtest)
    assert not prop_eq("a", "2")(dtest)
    assert not prop_eq("c", "1")(dtest)
    class __attrtest:
        def __init__(self):
            self.a: str = "a"
            self.one: int = 1
    attrtest = __attrtest()
    assert attr("a")(attrtest) == "a"
    assert attr("one")(attrtest) == 1
    assert attr("c")(attrtest) == None
    assert attr_or("a", "default")(attrtest) == "a"
    assert attr_or("c", "default")(attrtest) == "default"
    assert attr_eq("a", "a")(attrtest)
    assert attr_eq("one", 1)(attrtest)
    assert not attr_eq("a", "2")(attrtest)
    assert not attr_eq("c", "1")(attrtest)

    # String Functions
    assert split(" ", "split function test") == ["split", "function", "test"]
    assert replace(" ", "|", "replace function test") == "replace|function|test"

    # Math Functions
    assert add(1)(7) == 1 + 7
    assert mul(3)(7)   == 3 * 7
    assert sub_from(7)(3) == 7 - 3
    assert sub_this(3)(7) == 7 - 3
    assert div_this(8)(4) == 8 / 4
    assert div_by(4)(8)   == 8 / 4



