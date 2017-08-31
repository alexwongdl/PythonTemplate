"""
Created by Alex Wang
On 2017-08-31
"""
import requests

def image_download(url, save_path):
    """
    根据url下载图片并保存到save_path
    :param url:
    :param save_path:
    :return:
    """
    with open(save_path, 'wb') as handle:
        response = requests.get(url, stream=True)
        if not response.ok:
            return False

        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
        return True

if __name__ == "__main__":
    image_download('http://dmr.nosdn.127.net/v-20170826-6b05cdaa733282703f729b5afcc65759.jpg','E://temp/docduplicate/image/v-20170826-6b05cdaa733282703f729b5afcc65759.jpg')