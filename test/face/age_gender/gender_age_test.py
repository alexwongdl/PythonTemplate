"""
Created by Alex Wang
On 2018-06-26
"F" if predicted_genders[i][0] > 0.5 else "M"
"""
import os
import traceback
import time

import numpy as np
import cv2
from wide_resnet import WideResNet

face_size = 64
model = WideResNet(face_size, depth=16, k=8)()
model.load_weights('weights.18-4.06.hdf5')


def test_one(image_path):
    try:
        face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        img_org = cv2.imread(image_path)
        height, width = img_org.shape[0:2]
        img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, 1.1, 3)
        for (x, y, w, h) in faces:
            up_margin = w / 3
            margin = w / 4
            y_min = max(0, y - up_margin)
            y_max = min(height, y + h + margin)
            x_min = max(0, x - margin)
            x_max = min(width, x + w + margin)
            face_org = img_org[y_min:y_max, x_min: x_max, :].copy()

            # Change the image path with yours.
            # img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
            # x = image.img_to_array(img)

            face_img = cv2.resize(face_org, dsize=(face_size, face_size), interpolation=cv2.INTER_AREA)
            x = np.expand_dims(face_img, axis=0)
            print(x.dtype)
            start_time = time.time()
            results = model.predict(x)
            end_time = time.time()
            print('predict cost time:{}'.format(end_time - start_time))
            predicted_genders = results[0]
            gender = "F" if predicted_genders[0][0] > 0.5 else "M"
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            print(
            'gender score:{}, gender:{}, age:{:.2f}'.format(predicted_genders, gender, predicted_ages.tolist()[0]))
            # cv2.imshow('img_org', img_org)
            des_str = 'gender:{}, age:{:.2f}'.format(gender, predicted_ages.tolist()[0])
            cv2.putText(face_org, des_str, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        color=(0, 0, 255),
                        thickness=2)
            cv2.imshow('face', face_org)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        traceback.print_exc()
        print('error:{}'.format(image_path))


def test_batch():
    dir = 'data'
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        test_one(file_path)


def test_batch_1():
    data_root = 'SCUT-FBP5500_v2'
    image_root = os.path.join(data_root, 'Images')
    image_save_root = os.path.join('/Users/alexwang/data', 'gender_age')

    if not os.path.exists(image_save_root):
        os.mkdir(image_save_root)
    file_path = os.path.join(data_root, 'train_test_files', '5_folders_cross_validations_files',
                             'cross_validation_5', 'test_5.txt')
    image_list = {}
    with open(file_path, 'r') as reader:
        for line in reader:
            image_name, label = line.split(' ')
            image_list[image_name] = label.strip()
    for image_name in image_list.keys():
        try:
            image = cv2.imread(os.path.join(image_root, image_name))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_img = cv2.resize(image, dsize=(face_size, face_size), interpolation=cv2.INTER_AREA)
            x = np.expand_dims(face_img, axis=0)
            # print(x.dtype)
            start_time = time.time()
            results = model.predict(x)
            end_time = time.time()
            print('predict cost time:{}'.format(end_time - start_time))
            predicted_genders = results[0]
            gender = "F" if predicted_genders[0][0] > 0.5 else "M"
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            print(
            'gender score:{}, gender:{}, age:{:.2f}'.format(predicted_genders, gender, predicted_ages.tolist()[0]))

            des_str = 'gender:{}, age:{:.2f}'.format(gender, predicted_ages.tolist()[0])

            cv2.putText(image, des_str, (20, 20), cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        color=(0, 0, 255),
                        thickness=2
                        )

            save_path = os.path.join(image_save_root, '{}_{}'.
                                     format(gender, image_name))

            cv2.imwrite(save_path, image)
        except Exception as e:
            print('error with image:{}'.format(image_name))
            traceback.print_exc()


def statistics():
    totoal, MM, FF, FM, MF = 0, 0, 0, 0, 0
    image_save_root = os.path.join('data', 'gender_age')
    for file in os.listdir(image_save_root):
        if '_' in file:
            totoal += 1
            predict = file[0]
            label = file[3]
            if label == 'M':
                if predict == 'M':
                    MM += 1
                elif predict == 'F':
                    MF += 1
            elif label == 'F':
                if predict == 'F':
                    FF += 1
                elif predict == 'M':
                    FM += 1
    print('total:{}, MM:{}, FF:{}, MF:{}, FM:{}'.format(totoal, MM, FF, MF, FM))


if __name__ == '__main__':
    # test_one('data/face_one.png')
    # test_one('data/face_two.jpg')
    test_batch()
    # test_batch_1()
    statistics()
