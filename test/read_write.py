"""
Created by hzwangjian1
On 2017-08-31
"""
import traceback
def read_write(input_path, output_path, label):
    with open(output_path, 'w') as whandler:
        with open(input_path, 'r') as rhandler:
            str = rhandler.readline()
            while str:
                try:
                    sub_strs = str.split("\t")
                    whandler.write(sub_strs[0] + "\t" + sub_strs[1] + "\t" + label + "\n")
                    str = rhandler.readline()
                except Exception as e:
                    traceback.print_exc()

if __name__ == '__main__':
    read_write('E://temp/docduplicate/image/TrueNegative','E://temp/docduplicate/image/TrueNegative.format','0')
    read_write('E://temp/docduplicate/image/TruePositive','E://temp/docduplicate/image/TruePositive.format','1')