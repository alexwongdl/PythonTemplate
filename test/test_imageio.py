"""
Created by Alex Wang on 2018-03-02
测试ImageIO库  http://imageio.github.io/
包括ffmpeg在线读取视频内容
"""
import imageio
import cv2

def imageio_ffmpeg_video_url():
    """
    使用imageio和ffmpeg根据视频url获取视频的某一帧
    :return:
    """
    vid = imageio.get_reader('http://flv3.bn.netease.com/videolib3/1803/02/UARne7719/SD/UARne7719-mobile.mp4',  'ffmpeg')
    v_len = vid.get_length()
    frame = vid.get_data(int(v_len/2))
    print(dir(vid))

    cv2.imshow('mid_frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    imageio_ffmpeg_video_url()