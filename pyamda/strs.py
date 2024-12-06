from typing import List
from operator import methodcaller as method

from pyamda.core import FnU, partial


def replace(old: str, new: str) -> FnU[str, str]:
    """
    Curried, pure s.replace().

    >>> assert replace("x", "y")("x") == "x".replace("x", "y") == "y"
    """
    return method("replace", old, new)


def remove(*args: str) -> FnU[str, str]:
    """
    Replaces the given string with "" in the string fed to the resulting function.

    >>> assert remove("a")("abc") == "bc"
    >>> assert remove("b")("abc") == "ac"
    >>> assert remove("a", "b")("abc") == "c"
    >>> assert remove("x", "y")("xyz") == "z"
    """
    if isinstance(args, str):

        def remover(input_str: str) -> str:
            return input_str.replace(args, "")

    else:

        def remover(input_str: str) -> str:
            for substring in args:
                input_str = input_str.replace(substring, "")
            return input_str

    return remover


def concat(s: str) -> FnU[str, str]:
    """
    Returns a function that adds this string to the end of the argument to the resulting function.

    >>> assert concat("new")("old") == "oldnew"
    >>> assert concat("last")("first") == "firstlast"
    """
    return method("__add__", s)


def split(sep: str, maxsplits: int = -1) -> FnU[str, List[str]]:
    """
    Curried, pure s.split().

    >>> assert split("x")("1x2x3x4x5") == "1x2x3x4x5".split("x") == ["1","2","3","4","5"]
    >>> assert split("x", 2)("1x2x3x4x5") == "1x2x3x4x5".split("x", 2) == ["1","2","3x4x5"]
    """
    return method("split", sep, maxsplits)


def join_with(sep: str) -> FnU[List[str], str]:
    """
    Curried, pure s.join().

    >>> assert join_with(" ")(["1", "2", "3"]) == " ".join(["1", "2", "3"]) == "1 2 3"
    """
    return sep.join


def title() -> FnU[str, str]:
    """
    Curried, pure s.title().

    >>> assert title()("dog") == "dog".title() == "Dog"
    """
    return method("title")


def lower() -> FnU[str, str]:
    """
    Curried, pure s.lower().

    >>> assert lower()("DOg") == "DOg".lower() == "dog"
    """
    return method("lower")


def upper() -> FnU[str, str]:
    """
    Curried, pure s.upper().

    >>> assert upper()("DOg") == "DOg".upper() == "DOG"
    """
    return method("upper")
