def const(v):
    """ Build a function always return v
    :param v: bound value
    """
    return lambda _: v


def left_append(l):
    """ Build a function append l from left-hand
    """
    return lambda s: l + s


def right_append(r):
    """ Build a function append r from right-hand
    """
    return lambda s: s + r


def enclose_with(l, r=None):
    """ Build a function enclose with l and r
    """
    return lambda s: l + s + (r if r else l)