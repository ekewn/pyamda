from typing import Any, Optional

from core import op, partial, FnU


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
