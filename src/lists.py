from itertools import filterfalse
from typing import List

from core import FnU, in_, op, is_namedtuple


type NewList[a] = List[a]


def adjust[a](idx: int, fn: FnU[a, a], l: List[a]) -> NewList[a]:
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
    l2 = l.copy()
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

    def _(x, y):
        return op.attrgetter(y)(x) if is_namedtuple(x) else op.itemgetter(y)(x)

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
    return list(filterfalse(in_(items_to_remove), l2))


def startswith[a](val: a, l: List[a]) -> bool:
    """
    Does the list start with the given value?
    """
    return l[0] == val


def endswith[a](val: a, l: List[a]) -> bool:
    """
    Does the list end with the given value?
    """
    return l[len(l) - 1] == val
