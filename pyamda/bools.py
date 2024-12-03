from typing import Any, Iterable

from pyamda.core import Predicate, partial, op, flip


def T(*args) -> bool:
    """
    Always returns True.

    >>> assert T()
    >>> assert T(1)
    >>> assert T(False)
    """
    return True


def F(*args) -> bool:
    """
    Always returns False.

    >>> assert not F()
    >>> assert not F(1)
    >>> assert not F(True)
    """
    return False


def both[a](p1: Predicate[a], p2: Predicate[a]) -> Predicate[a]:
    """
    Returns a function that returns True if both of the predicates are true.

    >>> assert both(lambda x: x > 10, lambda x: x < 12)(11)
    >>> assert not both(lambda x: x > 10, lambda x: x < 12)(13)
    """

    def _(x, y, arg) -> bool:
        return x(arg) and y(arg)

    return partial(_, p1, p2)


def either[a](p1: Predicate[a], p2: Predicate[a]) -> Predicate[a]:
    """
    Returns a function that returns True if either of the predicates are true.

    >>> assert either(lambda x: x > 20, lambda x: x < 10)(30)
    >>> assert either(lambda x: x > 20, lambda x: x < 10)(0)
    >>> assert not either(lambda x: x > 20, lambda x: x < 10)(15)
    """

    def _(x, y, arg) -> bool:
        return x(arg) or y(arg)

    return partial(_, p1, p2)


def eq(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.eq.

    >>> assert eq(1)(1)
    >>> assert not eq(1)(2)
    """
    return partial(op.eq, x)


def gt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.gt.

    >>> assert gt(1)(3)
    >>> assert not gt(2)(1)
    """
    return partial(flip(op.gt), x)


def ge(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.ge.

    >>> assert ge(1)(1)
    >>> assert ge(0)(1)
    >>> assert not ge(2)(1)
    """
    return partial(flip(op.ge), x)


def lt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.lt.

    >>> assert lt(2)(1)
    >>> assert not lt(1)(1)
    """
    return partial(flip(op.lt), x)


def le(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.le.

    >>> assert le(1)(1)
    >>> assert le(2)(1)
    """
    return partial(flip(op.le), x)


def is_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.is_.

    >>> assert is_(None)(None)
    >>> assert not is_(None)({"x": 1})
    """
    return partial(flip(op.is_), x)


def is_not(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.is_not.

    >>> assert is_not(None)({"x": 1})
    >>> assert not is_not(None)(None)
    """
    return partial(op.is_not, x)


def and_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.and_.

    >>> assert and_(True)(True)
    >>> assert not and_(True)(False)
    """
    return partial(op.and_, x)


def or_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.or_.

    >>> assert or_(True)(False)
    >>> assert or_(True)(True)
    >>> assert not or_(False)(False)
    """
    return partial(op.or_, x)


def complement[a](p: Predicate[a]) -> Predicate[a]:
    """
    Returns a predicate that will return false when the given predicate would return true.

    >>> assert complement(lambda x: x == 0)(1)
    >>> assert not complement(lambda x: x == 0)(0)
    """

    def _(pred, val):
        return not pred(val)

    return partial(_, p)


def all_[a](p: Predicate[a]) -> Predicate[Iterable[a]]:
    """
    Returns a predicate that checks to see if the initial predicate holds for all items in the iterable.

    >>> l = [1, 2, 3]
    >>> p1 = lambda x : x > 0
    >>> p2 = lambda x : x > 1
    >>> assert all_(p1)(l)
    >>> assert not all_(p2)(l)
    """

    def _(pred, it):
        return all(map(pred, it))

    return partial(_, p)


def any_[a](p: Predicate[a]) -> Predicate[Iterable[a]]:
    """
    Returns a predicate that checks to see if the initial predicate holds for any items in the iterable.

    >>> l = [1, 2, 3]
    >>> p1 = lambda x : x > 2
    >>> p2 = lambda x : x > 3
    >>> assert any_(p1)(l)
    >>> assert not any_(p2)(l)
    """

    def _(pred, it):
        return any(map(pred, it))

    return partial(_, p)
