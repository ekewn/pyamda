# Class Instance Functions


#
#
# TESTS
#
#


if __name__ == "__main__":
    # Curried Built-ins
    assert list(take(3, map(add(1), count()))) == list(take(3, map_(add(1))(count())))
    assert list(take(3, filter(gt(2), count()))) == list(
        take(3, filter_(gt(2))(count()))
    )
    assert_(T)(1)

    # Composers
    assert (
        compose(len, add(10), sub_this(1))("number should be 28")
        == len("number should be 28") + 10 - 1
    )
    assert pipe(1, add(1), mul(3)) == (1 + 1) * 3
    assert foreach(print)("test") == "test"

    # Composition Helpers
    assert identity("test") == "test"
    assert tap(identity, "2") == "2"
    assert print_arg(10) == 10
    assert always("test")() == "test"
    assert tap(add(1), 1) == 1
    assert const(1)("anything") == 1
    assert const("this")(1) == "this"
    assert default_to(10, 11) == 11
    assert default_to(10, None) == 10
    assert default_with(10, add(1))(20) == 21
    assert default_with(10, none)(20) == 10

    # Logical
    assert T()
    assert not F()
    assert both(T, T)("anything")
    assert not both(T, F)("anything")
    assert not both(F, F)("anything")
    assert either(T, T)("anything")
    assert either(T, F)("anything")
    assert not either(F, F)("anything")
    assert not eq(1)(0)
    assert eq(1)(1)
    assert not eq(1)(2)
    assert not gt(1)(0)
    assert not gt(1)(1)
    assert gt(1)(2)
    assert not ge(1)(0)
    assert ge(1)(1)
    assert ge(1)(2)
    assert lt(1)(0)
    assert not lt(1)(1)
    assert not lt(1)(2)
    assert le(1)(0)
    assert le(1)(1)
    assert not le(1)(2)
    assert is_(1)(1)
    assert not is_(1)(2)
    assert not is_not(1)(1)
    assert is_not(1)(2)
    assert and_(True)(True)
    assert not and_(True)(False)
    assert or_(True)(False)
    assert not or_(False)(False)
    assert contains([0, 1, 2])(1)
    assert not contains([0, 1, 2])(3)
    assert in_(1)([0, 1, 2])
    assert not in_(3)([0, 1, 2])
    assert complement(is_none)(None) == False
    assert complement(complement(is_none))(None) == True

    # Control Flow
    assert if_else(T, const("a"), identity)("anything") == "a"
    assert if_else(F, const("a"), identity)("anything") == "anything"
    assert unless(T, add(1))(1) == 1
    assert unless(F, add(1))(1) == 2
    assert when(T, add(1))(1) == 2
    assert when(F, add(1))(1) == 1
    __condtest: FnU[int, str] = default_with(
        "otherwise",
        cond(
            [
                (gt(0), const("is positive")),
                (eq(0), const("is zero")),
                (lt(0), const("is negative")),
            ]
        ),
    )
    assert __condtest(1) == "is positive"
    assert __condtest(0) == "is zero"
    assert __condtest(-1) == "is negative"
    assert is_err(
        try_except(div_by(0), lambda v, e: Exception(f"arg: {v} caused err {e}"))(1)
    )
    assert is_err(try_(div_by(0))(1))
    assert optional(div_by(0))(1) is None
    assert optional(div_by(1))(1) == 1
    assert on_success(add(1))(1) == 2
    assert is_err(on_success(add(1))(Exception()))
    assert on_err(add(1))(1) == 1
    assert on_err(compose(err_val, add(1)))(Exception(1)) == 2

    # Container-related
    assert all(
        [
            is_empty(empty(["list"])),
            is_empty(empty({"dict": 1})),
            is_empty(empty(123)),
            is_empty(empty("string")),
        ]
    )
    assert not all(
        [
            is_empty(["this should fail because I'm not empty"]),
            is_empty({}),
            is_empty(0),
            is_empty(""),
        ]
    )
    assert all(
        [
            is_str("string"),
            is_list([1, 2]),
            is_bool(True),
            is_float(3.14),
            is_int(1),
            is_dict({"1": 1}),
        ]
    )
    assert is_none(None)
    assert not is_none("this should fail because I'm not None")
    assert is_err(Exception("this is err!"))
    assert not is_err(1)

    # Iterable Generics
    assert count_of(1)([1, 1, 0, 1]) == op.countOf([1, 1, 0, 1], 1)

    # Iterator Specifics
    assert list(take(4, iterate(add(3), 2))) == [2, 5, 8, 11]
    assert list(take(3, drop(2, count()))) == [2, 3, 4]
    assert head(count()) == 0
    assert list(take(3, tail(count()))) == [1, 2, 3]
    _partitiontest1, _partitiontest2 = partition(gt(3), take(6, count()))
    assert list(_partitiontest1) == [4, 5]
    assert list(_partitiontest2) == [0, 1, 2, 3]

    # List Functions
    assert adjust(2, add(3), [0, 1, 2, 3]) == [0, 1, 5, 3]
    assert move(0, 2, [0, 1, 2, 3, 4]) == [1, 2, 0, 3, 4]
    assert swap(0, 2, [0, 1, 2, 3, 4]) == [2, 1, 0, 3, 4]
    assert update(0, 2, [0, 1, 2, 3, 4]) == [2, 1, 2, 3, 4]
    assert cons(0, [0, 1, 2, 3, 4]) == [0, 0, 1, 2, 3, 4]
    assert cons([0, 1], [0, 1, 2, 3, 4]) == [0, 1, 0, 1, 2, 3, 4]
    assert startswith(0, [0, 1, 2])
    assert not startswith(0, [1, 2])
    assert endswith(2, [0, 1, 2])
    assert not endswith(2, [0, 1])
    assert remove(0, 2, [0, 1, 2, 3]) == [2, 3]
    assert not remove(0, 2, [0, 1, 2, 3]) == [1, 2, 3]
    assert pluck(0, [(0, 1), (0, 1)]) == [0, 0]
    assert pluck("a", [{"a": 0}, {"a": 0}]) == [0, 0]
    assert without([0, 1], [0, 2, 3, 1]) == [2, 3]

    # Dictionary Functions
    _dtest: Dict[str, str] = {"a": "1", "b": "2", "z": "3"}
    assert get(_dtest, "default", "a") == "1"
    assert get(_dtest, "default", "c") == "default"
    assert prop("a", _dtest) == "1"
    assert prop("c", _dtest) == None
    assert props(["a", "b", "c"], _dtest) == ["1", "2", None]
    assert prop_or("a", "default", _dtest) == "1"
    assert prop_or("c", "default", _dtest) == "default"
    assert prop_eq("a", "1", _dtest)
    assert not prop_eq("a", "2", _dtest)
    assert not prop_eq("c", "1", _dtest)
    assert prop_satisfies("a", lambda p: isinstance(p, str), _dtest)
    assert prop_satisfies("c", is_none, _dtest)
    assert project(["a", "b"], [_dtest, _dtest]) == [
        {"a": "1", "b": "2"},
        {"a": "1", "b": "2"},
    ]

    # Class Instance Functions
    class __attrtest:
        def __init__(self):
            self.a: str = "a"
            self.one: int = 1

    _attrtest = __attrtest()
    assert attr("a")(_attrtest) == "a"
    assert attr("one")(_attrtest) == 1
    assert attr("c")(_attrtest) == None
    assert attr_or("a", "default")(_attrtest) == "a"
    assert attr_or("c", "default")(_attrtest) == "default"
    assert attr_eq("a", "a")(_attrtest)
    assert attr_eq("one", 1)(_attrtest)
    assert not attr_eq("a", "2")(_attrtest)
    assert not attr_eq("c", "1")(_attrtest)

    # Math Functions
    assert add(1)(7) == 1 + 7
    assert mul(3)(7) == 3 * 7
    assert sub_from(7)(3) == 7 - 3
    assert sub_this(3)(7) == 7 - 3
    assert div_this(8)(4) == 8 / 4
    assert div_by(4)(8) == 8 / 4
    assert mod(3)(7) == 1
    assert round_to(1)(3.13) == 3.1

    assert print_("All tests passed!")("dog") == ("dog")
