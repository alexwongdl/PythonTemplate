# coding: utf-8

"""
Created on 2017-03-21
@author: timedcy@gmail.com
"""

from ddzj.util import cmd


def ffmpeg_get_keyframe(infilename, outfilename, timeout=None):
    """
    :param infilename: input video file name or url
    :param outfilename: output images file name with c print int format, like 'pics/thumbnails-%03d.jpeg'
    :param timeout: timeout in seconds, type: float
    :return:
    """
    retcode, stdout = cmd.run(
        ['ffmpeg', '-hide_banner', '-nostats', '-v', 'quiet', '-skip_frame', 'nokey', '-vsync', '0', '-t', '100', '-i', infilename,
         '-f', 'image2',
         outfilename],
        timeout=timeout)
    return retcode, stdout.decode() if retcode == 0 else ''

def _ffmpeg_get_audio(infilename, outfilename, timeout=None):
    """
    :param infilename: input video file name or url
    :param outfilename: output audio file name with c print int format, like 'test.wav'
    :param timeout: timeout in seconds, type: float
    :return:
    """
    retcode, stdout = cmd.run(
        ['ffmpeg', '-hide_banner', '-nostats', '-v', 'quiet', '-i',  infilename, '-f', 'wav', '-t', '10', '-vn', '-y', outfilename],
        timeout=timeout)
    return retcode, stdout.decode() if retcode == 0 else ''


def ffmpeg_get_audio(infilename, outfilename, timeout=None, n_try=4):
    """
    :param infilename: input video file name or url
    :param outfilename: output audio file name with c print int format, like 'pics/thumbnails-%03d.jpeg'
    :param timeout: timeout in seconds, type: float
    :param n_try: try count
    :return: done True or else False
    """
    if not infilename:
        return False
    now_try = 0
    timeout2 = timeout
    while 1:
        try:
            retcode, stdout = _ffmpeg_get_audio(infilename, outfilename, timeout2)
            if not retcode:
                return True
        except Exception:
            pass
        now_try += 1
        if now_try >= n_try or n_try is None:
            break
        if timeout2 is not None:
            timeout2 *= 2
    return False


def get_keyframes(infilename, outfilename, timeout=None, n_try=4):
    """
    :param infilename: input video file name or url
    :param outfilename: output images file name with c print int format, like 'pics/thumbnails-%03d.jpeg'
    :param timeout: timeout in seconds, type: float
    :param n_try: try count
    :return: done True or else False
    """
    if not infilename:
        return False
    now_try = 0
    timeout2 = timeout
    while 1:
        try:
            retcode, stdout = ffmpeg_get_keyframe(infilename, outfilename, timeout2)
            if not retcode:
                return True
        except Exception:
            pass
        now_try += 1
        if now_try >= n_try or n_try is None:
            break
        if timeout2 is not None:
            timeout2 *= 2
    return False
