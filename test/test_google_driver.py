"""
 @author: AlexWang
 @date: 2020/12/14 1:57 PM

 从google driver下载大文件
 参考:https://www.dazhuanlan.com/2019/10/05/5d97f9ea032b3/
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive

 share 链接:https://drive.google.com/file/d/FILEIDENTIFIER/view?usp=sharing
 #!/bin/bash
fileid="FILEIDENTIFIER"
filename="FILENAME"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
"""

import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    # https://drive.google.com/file/d/1CN_w8gR3nc4B9QNxpVx-GhgYYFA8bJs-/view?usp=sharing
    # https://drive.google.com/file/d/1Dh-leSxL2QDnHtXSxxQKtXlIyayCoH-n/view?usp=sharing
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)
    print("token:{}".format(token))

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


if __name__ == '__main__':
    download_file_from_google_drive("1Dh-leSxL2QDnHtXSxxQKtXlIyayCoH-n",
                                    "/Users/alexwang/TeamDrive/我的文件库/数据集/VCDB/02.zip")
