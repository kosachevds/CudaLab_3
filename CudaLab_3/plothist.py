import os
from matplotlib import pyplot as pp


def main():
    dirname = os.path.dirname(__file__)
    filename = "hist.txt"
    with open(os.path.join(dirname, filename), "rt") as hist_file:
        values = [int(x) for x in hist_file.read().split()]
    divider = sum(values)
    for x, y in enumerate(values):
        pp.bar(x, y / divider, width=1, color='b')
    pp.show()

if __name__ == '__main__':
    main()
