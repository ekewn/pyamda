from typing import Any, Optional

from core import op, partial, FnU


def attr(name: str) -> FnU[object, Optional[Any]]:
    """
    Returns a function that given an object returns the value of the object attribute if present, or none.

    >>> class Test:
    ...     one = 1
    >>> get_one = attr("one")
    >>> get_two = attr("two")
    >>> assert get_one(Test()) == 1
    >>> assert get_two(Test()) == None
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

    >>> class Test:
    ...     one = 1
    >>> get_one_or_0 = attr_or("one", 0)
    >>> get_two_or_12 = attr_or("two", 12)
    >>> assert get_one_or_0(Test()) == 1
    >>> assert get_two_or_12(Test()) == 12
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

    >>> class Test:
    ...     one = 1
    >>> one_is_1 = attr_eq("one", 1)
    >>> one_is_2 = attr_eq("two", 2)
    >>> assert one_is_1(Test())
    >>> assert not one_is_2(Test())
    """

    def _(n, v, o):
        try:
            return op.attrgetter(n)(o) == v
        except:
            return False

    return partial(_, name, val)
