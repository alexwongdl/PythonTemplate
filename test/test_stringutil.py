### Alex Wang @ 20170512
import sys
import os
# rootpath = os.path.dirname(os.getcwd())
# print(rootpath )
# myutil_path =  os.path.join(rootpath, "myutil")
# print(myutil_path)
# sys.path.append(myutil_path)

from myutil import stringutil

def test():
    print("test")
    print(stringutil.to_bytes("kdjfkd"))
    print(stringutil.to_str(b'dsfjksdl'))
