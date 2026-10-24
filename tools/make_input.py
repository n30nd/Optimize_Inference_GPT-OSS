#!/usr/bin/env python3
"""Writes a getp input file made of another input file's prompts repeated, for
throughput measurements at larger batches.

    tools/make_input.py evaluation/input.txt 8192 > input_8192.txt

The first lines of the output are the source prompts in their order, so the
first answers of a run can still be scored against the source's references.
"""

import sys


def main():
    src, count = sys.argv[1], int(sys.argv[2])
    with open(src) as f:
        f.readline()  # request count
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    print(count)
    for i in range(count):
        print(prompts[i % len(prompts)])


if __name__ == "__main__":
    main()
