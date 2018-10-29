import os
from matplotlib import pyplot as pp


def main():
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, "times2.txt")
    with open(filename, "rt") as times_file:
        text = times_file.read()
    parts = text.split(";")
    sizes = [int(x) for x in parts[0].split()]
    parts = parts[1:]
    for p in zip(parts, ["CPU", "GPU"]):
        pp.plot(sizes, [float(x) for x in p[0].split()], label=p[1])
    pp.ylabel("times, ms")
    pp.xlabel("sizes")
    pp.legend()
    pp.show()

if __name__ == '__main__':
    main()