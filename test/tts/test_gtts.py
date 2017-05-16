from test.tts.mytts import gTTS

def test():
    tts = gTTS("罗大姐说，她弟弟在买奔驰之前，就跟她提起过一个女朋友，按弟弟的描述，那就是一个典型的白富美，但弟弟从来没带对方来见过面",lang='zh')
    tts.save("E://temp/tts/gtts.mp3")
    # tts.save("/home/recsys/hzwangjian1/data/test_gtts91.mp3")