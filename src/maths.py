from core import FnU, op, partial, flip


def add[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.add. Returns unary function that adds this arg.

    >>> add1 = add(1)
    >>> assert add1(2) == 3
    >>> assert add1(4) == 5
    """
    return partial(op.add, arg)


def sub_from[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts from this arg.

    >>> sub_from10 = sub_from(10)
    >>> assert sub_from10(3) == 7
    >>> assert sub_from10(1) == 9
    """
    return partial(op.sub, arg)


def sub_this[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.sub. Returns unary function that subtracts this arg.

    >>> sub_this10 = sub_this(10)
    >>> assert sub_this10(3) == -7
    >>> assert sub_this10(1) == -9
    """
    return partial(flip(op.sub), arg)


def mul[a](arg: a) -> FnU[a, a]:
    """
    Curried operator.mul. Returns unary function that multiplies by this arg.

    >>> mul_by_2 = mul(2)
    >>> assert mul_by_2(10) == 20
    >>> assert mul_by_2(5) == 10
    """
    return partial(op.mul, arg)


def div_this[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the numerator as this arg.

    >>> div_12 = div_this(12)
    >>> assert div_12(3) == 4
    >>> assert div_12(6) == 2
    """
    return partial(op.floordiv, arg)


def div_by[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that sets the denominator as this arg.

    >>> div_by_12 = div_by(12)
    >>> assert div_by_12(24) == 2
    >>> assert div_by_12(48) == 4
    """
    return partial(flip(op.floordiv), arg)


def mod[a](arg: a) -> FnU[a, a]:
    """
    Curred operator.floordiv. Returns unary function that will perform modulo with this arg as right hand arg.

    >>> assert mod(3)(7) == 7 % 3 == 1
    >>> assert mod(7)(3) == 3 % 7 == 3
    """
    return partial(flip(op.mod), arg)


def round_to(num_digits: int) -> FnU[float, int | float]:
    """
    Curried round. Returns unary function that sets the denominator as this arg.

    >>> round_to_2 = round_to(2)
    >>> assert round_to_2(1.001) == 1.00
    >>> assert round_to_2(1.006) == 1.01
    """
    return partial(flip(round), num_digits)  # type: ignore
