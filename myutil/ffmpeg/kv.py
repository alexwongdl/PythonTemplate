"""
Created on 2017-02-23
@author: timedcy@gmail.com
"""


import collections


def flatten_dict(d, prefix='', sep='.'):
    items = []
    if isinstance(d, collections.MutableSequence):
        for idx, e in enumerate(d):
            prefix_1 = prefix + sep + str(idx) if prefix else str(idx)
            items.extend(flatten_dict(e, prefix_1, sep).items())
    elif isinstance(d, collections.MutableMapping):
        for k, v in d.items():
            prefix_2 = prefix + sep + k if prefix else k
            items.extend(flatten_dict(v, prefix_2, sep).items())
    else:
        items.append((prefix, d))
    return collections.OrderedDict(items)


def to_int(s, default_value=None):
    try:
        return int(s)
    except ValueError:
        return default_value


def to_float(s, default_value=None):
    try:
        return float(s)
    except ValueError:
        return default_value


def get_int(feat, key, default_value=0):
    ret = feat.get(key, default_value)
    if ret != default_value:
        ret = to_int(ret, default_value)
    return ret


def get_float(feat, key, default_value=0):
    ret = feat.get(key, default_value)
    if ret != default_value:
        ret = to_float(ret, default_value)
    return ret
