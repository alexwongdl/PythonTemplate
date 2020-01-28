"""
Created by Alex Wang on 2018-04-20
test functions in image_split.py
"""
import os
import traceback

import cv2

import image_split


def line_split():
    """
    :return:
    """
    root_dir = "/Users/alexwang/data/image_split"
    input_file = os.path.join(root_dir, 'image_split_test.txt')
    output_file = os.path.join(root_dir, 'image_split_test_norm.txt')

    with open(output_file, 'w') as writer:
        for line in open(input_file, 'r'):
            elems = line.split(" ")
            for elem in elems:
                elem = elem.strip()
                if elem.endswith('.jpg') or elem.endswith('.png'):
                    writer.write(elem + '\n')


def test_images_batch():
    """
    test a batch of images
    :return:
    """
    root_dir = "/Users/alexwang/data/image_split"
    input_dir = os.path.join(root_dir, 'image_split_test')
    output_dir = os.path.join(root_dir, 'image_split_result')

    process_index = 0
    start_index = 2000
    end_index = 3000

    for file in os.listdir(input_dir):

        try:
            if file.endswith('.jpg') or file.endswith('.png'):
                if process_index < start_index:
                    process_index += 1
                    continue

                print('process file:', file)
                comma_index = file.rfind('.')
                prefix = file[0:comma_index]
                postfix = file[comma_index:]
                img = cv2.imread(os.path.join(input_dir, file))
                image_patches = image_split.split_img(img, debug=False, plot=False)
                index = 1

                # if len(image_patches) > 0:  # save original image
                img_save_path = os.path.join(output_dir, file)
                cv2.imwrite(img_save_path, img)

                for image_patch in image_patches:
                    save_path = os.path.join(output_dir, "{}_{}{}".format(prefix, index, postfix))
                    print('save_path:', save_path)

                    cv2.imwrite(save_path, image_patch)
                    index += 1
            process_index += 1
            if process_index >= end_index:
                break

        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    test_images_batch()
    # line_split()
