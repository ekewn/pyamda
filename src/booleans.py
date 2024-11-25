from typing import Any

from core import Predicate, partial, op, flip

def T(*args) -> bool:
    """
    Always returns True.

    >>> assert(T() == True)
    >>> assert(T(1) == True)
    >>> assert(T(False) == True)
    """
    return True


def F(*args) -> bool:
    """
    Always returns False.

    >>> assert(F() == False)
    >>> assert(F(1) == False)
    >>> assert(F(True) == False)
    """
    return False


def both[a](p1: Predicate[a], p2: Predicate[a]) -> Predicate[a]:
    """
    Returns a function that returns True if both of the predicates are true.

    >>> assert(both(lambda x: x>10, lambda x: x <12)(11))
    >>> assert not (both(lambda x: x>10, lambda x: x <12)(13))
    """
    def _(x, y, arg) -> bool: return x(arg) and y(arg)
    return partial(_, p1, p2)


def either[a](p1: Predicate[a], p2: Predicate[a]) -> Predicate[a]:
    """
    Returns a function that returns True if either of the predicates are true.

    >>> assert(either(lambda x: x>10, lambda x: x <12)(13))
    >>> assert not (either(lambda x: x>10, lambda x: x <12)(11))
    """
    def _(x, y, arg) -> bool: return x(arg) or y(arg)
    return partial(_, p1, p2)


def eq(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.eq.

    >>> assert(eq(1)(1) == (1 == 1) == True)
    >>> assert(eq(1)(2) == (1 == 2) == False)
    """
    return partial(op.eq, x)


def gt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.gt.

    >>> assert(gt(1)(1) == (1 > 1) == False)
    >>> assert(gt(2)(1) == (2 > 1) == True)
    """
    return partial(flip(op.gt), x)


def ge(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.ge.

    >>> assert(ge(1)(1) == (1 >= 1) == True)
    >>> assert(ge(2)(1) == (2 >= 1) == True)
    """
    return partial(flip(op.ge), x)


def lt(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.lt.

    >>> assert(lt(1)(1) == (1 < 1) == False)
    >>> assert(lt(2)(1) == (2 < 1) == False)
    """
    return partial(flip(op.lt), x)


def le(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.le.

    >>> assert(le(1)(1) == (1 <= 1) == True)
    >>> assert(le(2)(1) == (2 <= 1) == False)
    """
    return partial(flip(op.le), x)


def is_(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.is_.

    >>> assert(is_(None)(None) == None is None == True)
    >>> assert(is_(1)(None) == 1 is None == False)
    >>> assert(is_(1)(1) == 1 is 1 == True)
    """
    return partial(flip(op.is_), x)


def is_not(x: Any) -> Predicate[Any]:
    """
    Curried version of operator.is_not.

    >>> assert(is_not(None)(None) == None is not None == False)
    >>> assert(is_not(1)(None) == 1 is not None == True)
    >>> assert(is_not(1)(1) == 1 is not 1 == False)
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


def complement[a](p: Predicate[a]) -> Predicate[a]:
    """
    Returns a predicate that will return false when the given predicate would return true.
    """
    def _(pred, val): return not pred(val)
    return partial(_, p)

