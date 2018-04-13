"""
Created by Alex Wang on 2018-04-13
pip install pyfiglet
"""
from pyfiglet import Figlet


def test_alex():
    text = 'Alex Wang'

    font_list = ['doom', 'drpepper', 'ntgreek', 'ogre', 'puffy', 'small', 'standard']

    for font_type in font_list:
        f = Figlet(font=font_type)
        print("Font:{:^10}:".format(font_type))
        print(f.renderText(text))


if __name__ == '__main__':
    test_alex()
