
import re
import sys

text_file = sys.argv[1]
keep_uttlist = sys.argv[2]

"""
1. filled pauses, UM, ER, UH, AH, and HMM, keep them
2. non-speech events: remove them, see nonspeech events
3. Truncated words: remove them
4. Unitelligible () remove
"""
nonspeech_events = ["<BREATH>", "<LAUGH>", "<COUGH>", "<NOISE>", "<SIDE_SPEECH>", "<NO_SIGNAL>", "<SILENCE>", "<SNIFF>", "<ECHO>", "<DISCARD>"]

filter_symbols = ["[", "]", "<", ">", "-", "…", "(", ")"]

def main():
    with open(text_file, "r") as rf, open(keep_uttlist, "w") as wf:
        line = rf.readline()
        n_filtered = 0
        while line:
            line = line.strip().split(' ')
            if len(line) == 1:
                line = rf.readline()
                continue

            utt, sent = line[0], line[1:]
            new_sent = []
            n_words = 0

            for word in sent:
                if len(word) == 0 or word in ["_", "+", "*", "/"]:
                    continue
                if word[0] in filter_symbols or word[-1] in filter_symbols or word in ["M&MS"] or word[-3:] == "<NO":
                    new_sent.append("<UNK>")              
                elif word[-1] in [".", "?", ";", ":"]:
                    new_sent.append(word[:-1])
                    n_words += 1
                else:
                    new_sent.append(word)
                    n_words += 1
            
            if n_words > 0:
                new_sent = " ".join(new_sent).replace("‘", "'").replace("’", "'")
                new_sent = re.sub(r"[/+:*0-9]", "", new_sent).split()
                if len(new_sent) > 0:
                    wf.write(utt + " " + " ".join(new_sent) + "\n")
            else:
                n_filtered += 1
                print("filter utterance:{}".format(utt))
            line = rf.readline()
        print("{} utterances are filtered!".format(n_filtered))

if __name__ == "__main__":
    main()


