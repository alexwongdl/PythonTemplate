# coding: utf-8

"""
Created on 2017-02-24
@author: timedcy@gmail.com
"""

import collections
import json
import subprocess

from ddzj.util import cmd
from ddzj.util.kv import flatten_dict


def ffprobe_show(filename):
    return subprocess.getstatusoutput(
        'ffprobe -v quiet -show_format -show_streams -print_format json "{}"'.format(filename))


def get_flatten_info(filename):
    info = ffprobe_show(filename)[1]
    result = json.loads(info)
    return flatten_dict(result)


def _get_info(filename, timeout=None):
    retcode, stdout = cmd.run(
        ['ffprobe', '-v', 'quiet', '-show_format', '-show_streams', '-print_format', 'json', filename],
        timeout=timeout)
    result = json.loads(stdout.decode()) if retcode == 0 and len(stdout) > 0 else {}
    return flatten_dict(result)


def get_info(filename, timeout=None, n_try=4):
    if not filename:
        return collections.OrderedDict()
    now_try = 0
    timeout2 = timeout
    while 1:
        try:
            ret = _get_info(filename, timeout2)
            if ret is not None and len(ret) > 0:
                return ret
        except Exception:
            pass
        now_try += 1
        if now_try >= n_try or n_try is None:
            break
        if timeout2 is not None:
            timeout2 *= timeout2
    return collections.OrderedDict()
