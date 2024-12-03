from collections import deque
from itertools import islice, repeat, accumulate, tee, filterfalse
from typing import Iterable, Iterator, Tuple

from pyamda.core import FnU, op, flip, partial, Predicate


def count_of(x: object) -> FnU[Iterable[object], int]:
    """
    Curried version of operator.countOf.

    >>> count_ones = count_of(1)
    >>> assert count_ones([1, 1, 1, 2, 3]) == 3
    """
    return partial(flip(op.countOf), x)


def consume(i: Iterator) -> None:
    """
    Consumes an iterable to trigger side effects (avoids wasting the creation of a list).
    Taken from python recipes. Faster than list comprehension just to trigger side effects.
    """
    deque(i, maxlen=0)


def take[a](n: int) -> FnU[Iterable[a], Iterator[a]]:
    """
    Returns an iterator of the first n items from the supplied iterable.

    >>> i = [0, 1, 2, 3]
    >>> first_2 = take(2)
    >>> assert list(first_2(i)) == [0, 1]
    """
    return lambda i: islice(i, n)


def drop[a](n: int) -> FnU[Iterable[a], Iterator[a]]:
    """
    Drops the first n items from an iterator.

    >>> i = [0, 1, 2, 3]
    >>> drop_2 = drop(2)
    >>> assert list(drop_2(i)) == [2, 3]
    """
    return lambda i: islice(i, n, None)


def head[a](i: Iterator[a]) -> a:
    """
    Gets first item from an iterator.

    >>> i = [0, 1, 2, 3]
    >>> first_2_items = take(2)(i)
    >>> assert head(first_2_items) == 0
    """
    return next(i)


def tail[a](i: Iterator[a]) -> Iterator[a]:
    """
    Returns an iterator without the first element of the given iterator.

    >>> i = [0, 1, 2, 3]
    >>> first_2_items = take(2)(i)
    >>> assert next(tail(first_2_items)) == 1
    """
    return drop(1)(i)


def iterate[a](fn: FnU[a, a]) -> FnU[a, Iterator[a]]:
    """
    Creates an iterator by applying the same function to the result of f(x).

    >>> times2 = lambda x : x * 2
    >>> repeatedly_multiply_by_2 = iterate(times2)
    >>> all_multiples_of_2 = repeatedly_multiply_by_2(2)
    >>> take3 = take(3)
    >>> assert list(take3(all_multiples_of_2)) == [2, 4, 8]
    """
    return lambda x: accumulate(repeat(x), lambda fx, _: fn(fx))


def partition[a](p: Predicate[a], i: Iterable[a]) -> Tuple[Iterator[a], Iterator[a]]:
    """
    Returns the iterable separated into those that satisfy and don't satisfy the predicate.

    >>> l = [0, 1, 2, 3, 4, 5, 6]
    >>> is_gt3 = lambda x : x > 3
    >>> gt3, le3 = partition(is_gt3, l)
    >>> assert list(gt3) == [4, 5, 6]
    >>> assert list(le3) == [0, 1, 2, 3]
    """
    t1, t2 = tee(i)
    return filter(p, t1), filterfalse(p, t2)
