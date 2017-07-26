"""
Created by Alex Wang
on 2017-07-26
"""

def test_dict_update():
    info = dict({"name": "Alex Wang", "age": 18})
    info.update({"gender": "male"})
    print(info)

if __name__ == "__main__" :
    test_dict_update()
