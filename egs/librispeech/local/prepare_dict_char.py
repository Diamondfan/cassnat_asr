
import os
import sys

dict_path = sys.argv[1]
train_text = sys.argv[2]
apply_text = sys.argv[3]

if not os.path.exists(dict_path):
    chars = ["|"]
    with open(train_text, 'r') as rf:
        line = rf.readline()
        while line:
            line = line.strip().split(" ")[1:]
            for word in line:
                word = list(word)
                chars.extend(word)
            line = rf.readline()
        chars = set(chars)
    with open(dict_path, 'w') as wf:
        for char in chars:
            wf.write(char + '\n')

with open(apply_text, 'r') as rf:
    line = rf.readline()
    while line:
        utt_id, text = line.strip().split(" ", 1)
        token = text.replace(" ", "|")
        token = " ".join(list(token))
        print(utt_id + " " + token, flush=True)
        line = rf.readline()

