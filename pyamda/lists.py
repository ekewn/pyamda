from typing import List, Dict, NamedTuple

from .core import FnU, Predicate, op, is_namedtuple, partial


def cons[a](val: a | List[a]) -> FnU[List[a], List[a]]:
    """
    Returns a copy of the list with the value/other list prepended.

    >>> l = [1, 2]
    >>> assert cons(0)(l) == [0, 1, 2]
    >>> assert cons([-1, 0])(l) == [-1, 0, 1, 2]

    """

    def _(v: a | List[a], l: List[a]) -> List[a]:
        if isinstance(v, List):
            assert isinstance(v, List)
            return v + l
        else:
            l2 = l.copy()
            l2.insert(0, v)
            return l2

    return partial(_, val)


def pluck[a: List, Dict, object](name_idx: int | str) -> FnU[List[a], List[a]]:
    """
    Returns a copy of the list by plucking a property (if given a property name) or an item (if given an index)
    off of each object/item in the list.

    >>> ldict = [{"a" : "firsta", "b": "firstb"}, {"a": "seconda", "b": "secondb"}]
    >>> llist = [["l1first", "l1second"], ["l2first", "l2second"]]
    >>> nt = NamedTuple("nt", [("a", str), ("b", str)])
    >>> lnt = [nt("nt1a", "nt1b"), nt("nt2a", "nt2b")]
    >>> assert pluck("a")(ldict) == ["firsta", "seconda"]
    >>> assert pluck(0)(llist) == ["l1first", "l2first"]
    >>> assert pluck("a")(lnt) == ["nt1a", "nt2a"]
    """

    def _(x: int | str, l: List[a]) -> List[a]:
        y = l[0]
        if isinstance(x, str) and (
            is_namedtuple(y) and not isinstance(y, dict) and not isinstance(y, list)
        ):
            assert isinstance(x, str)
            return [op.attrgetter(x)(_) for _ in l]
        else:
            return [op.itemgetter(x)(_) for _ in l]

    return partial(_, name_idx)


def without[a](items_to_remove: List[a]) -> FnU[List[a], List[a]]:
    """
    Returns a copy of the list with all the items from the first list (items to remove) taken out of the given list.

    >>> l = [0, 1, 2, 3, 4, 5]
    >>> assert without([0, 2])(l) == [1, 3, 4, 5]
    """

    def _(n: List[a], l: List[a]) -> List[a]:
        return [x for x in l if x not in n]

    return partial(_, items_to_remove)


def startswith[a](val: a) -> Predicate[List[a]]:
    """
    Returns a function that determines if the given list beings with the value given.

    >>> l = [0, 1, 2, 3]
    >>> assert startswith(0)(l)
    >>> assert not startswith(1)(l)
    """

    def _(v: a, l: List[a]) -> bool:
        return l[0] == v

    return partial(_, val)


def endswith[a](val: a) -> Predicate[List[a]]:
    """
    Does the list end with the given value?

    >>> l = [0, 1, 2, 3]
    >>> assert endswith(3)(l)
    >>> assert not endswith(2)(l)
    """

    def _(v: a, l: List[a]) -> bool:
        return l[-1] == v

    return partial(_, val)
