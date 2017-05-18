'''
Created on 2017-5-18
@author: Alex Wang
'''
def devide(a, b):
    try:
        return a/b
    except Exception as e:
        print(e.__str__())
        return None

def test_none():
    if devide(0,2) is None:
        print(devide(0,2))
        print("devide(0,2) is None")

    if devide(2,0) is None:
        print("devide(2,0) is None")



if __name__ == "__main__":
    test_none()
    if 0 is None:
        print("0 is None")