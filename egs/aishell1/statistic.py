
import kaldiio

feats = 'data/train_all/feats.scp'
label = 'data/train_all/token.scp'

feats_dict = {}
label_dict = {}

with open(feats, 'r') as rf1, open(label, 'r') as rf2:
    line = rf1.readline()
    while line:
        line = line.strip().split(' ', 1)
        feats_dict[line[0]] = line[1]
        line = rf1.readline()

    line = rf2.readline()
    while line:
        line = line.strip().split(' ', 1)
        label_dict[line[0]] = line[1]
        line = rf2.readline()

max_ratio = 0
i = 0
for utt in feats_dict:
    feats = kaldiio.load_mat(feats_dict[utt])
    frames = feats.shape[0]
    n_labels = len(label_dict[utt])
    ratio = n_labels / frames
    max_ratio = max(max_ratio, ratio)
    if i % 5000 == 0:
        print(i, n_labels, frames)
    i += 1
print(max_ratio)
   
