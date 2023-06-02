
import re
import sys

old_text = sys.argv[1]
new_text = sys.argv[2]

"""
1. filled pauses, UM, ER, UH, AH, and HMM, MMM, remove them
"""

filter_pauses = ["UM", "ER", "UH", "AH", "HMM", "MMM"]

def main():
    with open(old_text, "r") as rf, open(new_text, "w") as wf:
        line = rf.readline()
        while line:
            line = line.strip().split(' ')
            sent = line[:-1]
            new_sent = []
            for word in sent:
                if word not in filter_pauses:
                    new_sent.append(word)
            wf.write(" ".join(new_sent) + " " + line[-1] + "\n")
            line = rf.readline()
    print("Finish removing filter pauses in {}".format(old_text))

if __name__ == "__main__":
    main()


