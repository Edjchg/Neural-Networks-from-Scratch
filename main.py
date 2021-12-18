import sys
import matplotlib
import numpy as np


def show_information():
    info = "------------------------------------------------\n"
    info += "General Information:\n"
    info += "Description:\tNeural Networks from Scratch.\n"
    info += "Author:\tEdgar Chaves\n"
    info += "Year:\t2021-2022\n"
    info += "------------------------------------------------\n"
    print(info)


def show_references():
    references = "------------------------------------------------\n"
    references += "References: \n"
    references += "Sentdex: \thttps://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3" \
                  "&index=1 \n "
    references += "------------------------------------------------\n"
    print(references)


def show_libraries():
    libraries = "------------------------------------------------\n"
    libraries += "Used libraries:\n"
    libraries += "Python:" + "\t" + str(sys.version) + "\n"
    libraries += "Numpy:" + "\t" + str(np.__version__) + "\n"
    libraries += "Matplotlib:" + "\t" + matplotlib.__version__ + "\n"
    libraries += "------------------------------------------------\n"
    print(libraries)


if __name__ == '__main__':
    show_information()
    show_libraries()
    show_references()
