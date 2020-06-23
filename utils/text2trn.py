#!/usr/bin/env python3
# 2020 Ruchao Fan

import sys

text = sys.argv[1]
trn = sys.argv[2]


with open(text, 'r') as rf, open(trn, 'w') as wf:
    line = rf.readline()
    while line:
        line = line.strip().split(' ', 1)
        wf.write(line[1] + ' (' + line[0].replace('-', '_') + ')\n' )
        line = rf.readline()

