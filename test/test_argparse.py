"""
Created by Alex Wang
On 2017-10-15
"""
import argparse
from myutil.myprint import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='mehod_one', help='method type', type= str)
    parser.add_argument('-n', '--times', default='1', help='run times', type=int)

    args = parser.parse_args()
    # print(args.type)
    # print("times:" + str(args.times + 5))
    
    arg_parse_print(args)
