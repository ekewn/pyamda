import operator as op
from collections import deque
from functools import partial, reduce
from itertools import (accumulate, count, filterfalse, islice, repeat,
                       tee)
from typing import (Any, Callable, Container, Dict, Iterable, Iterator, List, NamedTuple,
                    Optional, Tuple)

#
#
# DATA
#
#




# Composers

# Logical


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
        if v is not None: return fn(v)
        else: return v
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
    with the arg nested in the exception i.e. you can retrieve it by doing err_val(Exception).
    """
    def _(t, v):
        try: return t(v)
        except Exception as err: return Exception(v, err)
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


def test[a, b](fn: FnU[a, b], cases: Iterable[Case[a, b]]) -> None:
    """
    Similar to apply_spec, where we apply the function to each "input" of the case and map the value to the output.
    Assert every application of the funtion to an input results in the expected output.
    """
    for case in cases:
        assert case.expected(fn(case.input)), f"{fn.__name__} - {case.name} - Expected {case.expected} but got {fn(case.input)}."



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


is_none: Predicate[Any] = is_(None)
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
    return isinstance(x, tuple) and hasattr(x, '_asdict') and hasattr(x, '_fields')


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
    def _(x, y): return x in y
    return partial(_, x) 



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


def pluck(name_idx: int | str, l: List) -> NewList:
    """
    Returns a copy of the list by plucking a property (if given a property name) or an item (if given an index)
    off of each object/item in the list.
    """
    l2 = l.copy()
    def _(x, y): return op.attrgetter(y)(x) if is_namedtuple(x) else op.itemgetter(y)(x)
    return [_(x, name_idx) for x in l2]


def remove[a](first: int, last: int, l: List[a]) -> NewList[a]:
    """
    Returns a copy of the list with all items from first to last indices given (not including the value at the last index) removed.
    """
    l2 = l.copy()
    del l2[first:last]
    return l2


def without[a](items_to_remove: List[a], l: List[a]) -> List[a]:
    """
    Returns a copy of the list with all the items from the first list (items to remove) taken out of the given list.
    """
    l2 = l.copy()
    return list(filterfalse(contains(items_to_remove), l2))


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


# Class Instance Functions

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


def mod[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that will perform modulo with this arg as right hand arg.
    e.g. mod(3)(7) == 7 % 3 == 0
    """
    return partial(flip(op.mod), arg)


def round_to(num_digits: int ) -> FnU[float, int | float]:
    """
    Curred round. Returns unary function that sets the denominator as this arg.
    e.g. mod(3)(7) == 7 % 3 == 0
    """
    return partial(flip(round), num_digits) #type: ignore


#
#
# TESTS
#
#


if __name__ == "__main__":
    # Curried Built-ins
    assert list(take(3, map(add(1), count()))) == list(take(3, map_(add(1))(count())))
    assert list(take(3, filter(gt(2), count())))    == list(take(3,filter_(gt(2))(count())))
    assert_(T)(1)


    # Composers
    assert compose(len, add(10), sub_this(1))("number should be 28") == len("number should be 28") + 10 - 1
    assert pipe(1, add(1), mul(3))                                == (1 + 1) * 3
    assert foreach(print)("test") == "test"

    # Composition Helpers
    assert identity("test")                        == "test"
    assert tap(identity, "2")                      == "2"
    assert print_arg(10) == 10
    assert always("test")()                        == "test"
    assert tap(add(1), 1)                          == 1
    assert const       (1)            ("anything") == 1
    assert const       ("this")       (1)          == "this"
    assert default_to  (10, 11)                    == 11
    assert default_to  (10, None)                  == 10
    assert default_with(10, add  (1))        (20)  == 21
    assert default_with(10, none)     (20)         == 10

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
    assert in_(1)([0, 1, 2])
    assert not in_(3)([0, 1, 2])
    assert complement(is_none)(None)             == False
    assert complement(complement(is_none))(None) == True

    # Control Flow
    assert if_else     (T, const      ("a"), identity)("anything") == "a"
    assert if_else     (F, const      ("a"), identity)("anything") == "anything"
    assert unless      (T, add   (1))        (1)                   == 1
    assert unless      (F, add   (1))        (1)                   == 2
    assert when        (T, add   (1))        (1)                   == 2
    assert when        (F, add   (1))        (1)                   == 1
    __condtest: FnU[int, str]                                        =  default_with("otherwise", cond([(gt(0), const("is positive"))
                                                                                                    , (eq(0), const("is zero"))
                                                                                                    , (lt(0), const("is negative"))]))
    assert __condtest(1)                                    == "is positive"
    assert __condtest(0)                                    == "is zero"
    assert __condtest(-1)                                   == "is negative"
    assert is_err(try_except(div_by(0), lambda v, e: Exception(f"arg: {v} caused err {e}"))(1))
    assert is_err(try_(div_by(0))(1))
    assert optional(div_by(0))(1) is None
    assert optional(div_by(1))(1)                         == 1
    assert on_success(add(1))(1)                          == 2
    assert is_err(on_success(add(1))(Exception()))
    assert on_err(add(1))(1)                              == 1
    assert on_err(compose(err_val, add(1)))(Exception(1)) == 2

    # Container-related
    assert all([is_empty(empty(["list"]))
               , is_empty(empty({"dict" : 1}))
               , is_empty(empty(123))
               , is_empty(empty("string"))])
    assert not all([is_empty(["this should fail because I'm not empty"])
               , is_empty({})
               , is_empty(0)
               , is_empty("")])
    assert all([is_str("string")
                , is_list([1, 2])
                , is_bool(True)
                , is_float(3.14)
                , is_int(1)
                , is_dict({"1" : 1})])
    assert is_none(None)
    assert not is_none("this should fail because I'm not None")
    assert is_err(Exception("this is err!"))
    assert not is_err(1)

    # Iterable Generics
    assert count_of(1)([1, 1, 0, 1]) == op.countOf([1, 1, 0, 1], 1)

    # Iterator Specifics
    assert list(take(4, iterate(add(3), 2))) == [2, 5, 8, 11]
    assert list(take(3, drop(2, count())))   == [2, 3, 4]
    assert head(count())                     == 0
    assert list(take(3, tail(count())))      == [1, 2, 3]
    _partitiontest1, _partitiontest2           =  partition(gt(3), take(6, count()))
    assert list(_partitiontest1)              == [4, 5]
    assert list(_partitiontest2)              == [0, 1, 2, 3]

    # List Functions
    assert adjust(2, add(3), [0, 1, 2, 3])  == [0, 1, 5, 3]
    assert move(0, 2, [0, 1, 2, 3, 4])      == [1, 2, 0, 3, 4]
    assert swap(0, 2, [0, 1, 2, 3, 4])      == [2, 1, 0, 3, 4]
    assert update(0, 2, [0, 1, 2, 3, 4])    == [2, 1, 2, 3, 4]
    assert cons(0, [0, 1, 2, 3, 4])         == [0, 0, 1, 2, 3, 4]
    assert cons([0, 1], [0, 1, 2, 3, 4])    == [0, 1, 0, 1, 2, 3, 4]
    assert startswith(0, [0, 1, 2])
    assert not startswith(0, [1, 2])
    assert endswith(2, [0, 1, 2])
    assert not endswith(2, [0, 1])
    assert remove(0, 2, [0, 1, 2, 3])       == [2, 3]
    assert not remove(0, 2, [0, 1, 2, 3])   == [1, 2, 3]
    assert pluck(0, [(0, 1), (0, 1)])       == [0, 0]
    assert pluck("a", [{"a": 0}, {"a": 0}]) == [0, 0]
    assert without([0, 1], [0, 2, 3, 1])    == [2, 3]

    # Dictionary Functions
    _dtest: Dict[str, str]                 =  {"a": "1", "b": "2", "z" : "3"}
    assert get(_dtest, "default", "a")     == "1"
    assert get(_dtest, "default", "c")     == "default"
    assert prop("a", _dtest)               == "1"
    assert prop("c", _dtest)               == None
    assert props(["a", "b", "c"], _dtest)  == ["1", "2", None]
    assert prop_or("a", "default", _dtest) == "1"
    assert prop_or("c", "default", _dtest) == "default"
    assert prop_eq("a", "1", _dtest)
    assert not prop_eq("a", "2", _dtest)
    assert not prop_eq("c", "1", _dtest)
    assert prop_satisfies("a", lambda p: isinstance(p, str), _dtest)
    assert prop_satisfies("c", is_none, _dtest)
    assert project(["a", "b"], [_dtest, _dtest]) == [{"a" : "1", "b" : "2"}, {"a" : "1", "b" : "2"}]

    # Class Instance Functions
    class __attrtest:
        def __init__(self):
            self.a: str = "a"
            self.one: int = 1
    _attrtest = __attrtest()
    assert attr("a")(_attrtest)               == "a"
    assert attr("one")(_attrtest)             == 1
    assert attr("c")(_attrtest)               == None
    assert attr_or("a", "default")(_attrtest) == "a"
    assert attr_or("c", "default")(_attrtest) == "default"
    assert attr_eq("a", "a")(_attrtest)
    assert attr_eq("one", 1)(_attrtest)
    assert not attr_eq("a", "2")(_attrtest)
    assert not attr_eq("c", "1")(_attrtest)

    # Math Functions
    assert add(1)(7)         == 1 + 7
    assert mul(3)(7)         == 3 * 7
    assert sub_from(7)(3)    == 7 - 3
    assert sub_this(3)(7)    == 7 - 3
    assert div_this(8)(4)    == 8 / 4
    assert div_by(4)(8)      == 8 / 4
    assert mod(3)(7)         == 1
    assert round_to(1)(3.13) == 3.1

    assert print_("All tests passed!")("dog") == ("dog")
