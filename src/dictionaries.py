from functools import partial as p
from typing import Dict, Optional, Any, List
from operator import methodcaller as method

from pyamda import FnU


def get[a, b](key: a) -> FnU[Dict[a,b], Optional[b]]:
    """
    Curried dict.get alias.

    >>> assert(get("a")({"a": 1}) == {"a": 1}.get("a") == 1)
    >>> assert(get("b")({"a": 1}) == {"a": 1}.get("b") == None)
    >>> assert(get("b", "dog")({"a": 1}) == {"a": 1}.get("b", "dog") == "dog")
    """
    return method("get", key)


def prop(key: str) -> FnU[Dict[str, Any], Optional[Any]]:
    """
    Returns a function that recursively checks the object to return the value at the key, if present, else None.
    """
    def _(key, d) -> Optional[Any]:
        if key in d:
            return d[key]
        else:
            for v in list(filter(is_dict, d.values())):
                return D.prop(key)(v) #type: ignore
    return p(_, key)


def prop_or[a](key: str, default: a) -> FnU[Dict[str, a], a]:
    """
    Returns a function that, given a dict, returns the default value if key is not found in the object.
    """
    return default_with(default, prop(key))


def pick(keys: List[str]) -> FnU[Dict[str, Any], Dict[str, Any]]:
    """
    Returns partial copies of the given dictionary with only the specified keys.
    Any keys that don't exist in the dictionary will not be included in the copy,
    i.e. they'll raise a KeyError if you try to attempt them.
    """
    def _(keys, d): return {k: v for k, v in d.items() if k in keys}
    return p(_, keys)


def omit(keys: List[str]) -> FnU[Dict[str, Any], Dict[str, Any]]:
    """
    Returns partial copies of the given dictionary with the specified keys dropped.
    """
    def _ (keys, d): return {k: v for k, v in d.items() if k not in keys}
    return p(_, keys)


def project(keys: List[str], ds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Think of this as an SQL select query. Gets the props off of each dict in the list.
    """
    return list(map(pick(keys), ds))


def apply_spec[a, b](fn: FnU, d: Dict[str, a]) -> Dict[str, b]: #type: ignore - b only shows up once in signature
    """
    Applies the provided function to each value in the provided dictionary.
    """
    return {k: fn(v) for k, v in d}
