"""
Created by Alex Wang
On 2018-06-06
ref: http://www.bogotobogo.com/VideoStreaming/YouTube/youtube-dl-embedding.php
"""
import traceback

import youtube_dl

def video_download_func(url, save_dir):
    """
    download video by url
    :param url:
    :param save_dir:
    :return:
    """
    try:
        ydl_opts = {'outtmpl': '{}/%(id)s'.format(save_dir)}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print('download failed:', url)
        traceback.print_exc()
        return False

def download_videos(video_save_dir, video_url_list):
    for url in video_url_list:
        video_download_func(url, video_save_dir)

if __name__ == '__main__':
    video_save_dir = '/Users/alexwang/data/activitynet'
    video_url_list = [
        'https://www.youtube.com/watch?v=sJFgo9H6zNo',
        'https://www.youtube.com/watch?v=V1zhqaGFY2A',
        'https://www.youtube.com/watch?v=JDg--pjY5gg'
    ]
    download_videos(video_save_dir, video_url_list)