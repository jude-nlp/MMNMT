import random
import argparse

def data_concate(src_path, tgt_path, max_cut_length):
    src = []
    tgt = []
    src_concat = []
    # load data from file
    with open(src_path, 'r', encoding='utf-8') as src_file:
        for line in src_file.readlines():
            src.append(line.strip())
    with open(tgt_path, 'r', encoding='utf-8') as tgt_file:
        for line in tgt_file.readlines():
            tgt.append(line.strip())
    length = len(src)
    assert len(src) == len(tgt)

    # ramdom choose data from target line
    for i in range(length):
        s = src[i].split()
        t = tgt[i].split()
        # the length of cut
        cut_length = random.randint(0, max_cut_length)

        # target length is small than cut_length
        if len(t) - 1 - cut_length < 0:
            s.extend(t)
            src_concat.append(s)
            continue

        # cut_length = 0
        if cut_length == 0:
            src_concat.append(s)
            continue

        # cut start pos
        pos = random.randint(0, len(t) - 1 - cut_length)
        frag = t[pos: pos + cut_length]
        s.extend(frag)
        src_concat.append(s)

    with open('src_concate.txt', 'w') as f:
        for line in src_concat:
            line = " ".join(line)
            f.write(line + '\n')


parser = argparse.ArgumentParser(description='copynet data processed')
parser.add_argument('--src_path', type=str, help='max_cut_length')
parser.add_argument('--tgt_path', type=str, help='max_cut_length')
parser.add_argument('--max_cut_length', type=int, default=3, help='max_cut_length')

args = parser.parse_args()

data_concate(args.src_path, args.tgt_path, max_cut_length=args.max_cut_length)