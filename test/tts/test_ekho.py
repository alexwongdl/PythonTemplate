import os
import subprocess

def call_ekho(text, audio_path):
    cmd_str = 'ekho {} -o {} -v Mandarin'.format(text, audio_path)
    d = subprocess.run(cmd_str, stdout=subprocess.PIPE, timeout=100, shell = True)
    pid = os.getpid()
    return d

def test():
    text = "安徽的罗大姐说，她有个27岁的弟弟，在杭州余杭良渚打工，10号那天，弟弟突然精神失常，与此同时，弟弟刚买不久的一辆奔驰也失踪了。"
    audio_path = '/home/recsys/hzwangjian1/tts/test_ekho_py.wav'
    call_ekho(text, audio_path)

if __name__ == "__main__":
    test()