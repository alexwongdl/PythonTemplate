"""
Created by Alex Wang
On 2018-07-23
"""
import cv2

def test_video_read_and_write():
    cap = cv2.VideoCapture('input_video.mp4')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # tps is 20
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width, frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    test_video_read_and_write()