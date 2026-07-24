import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file_path')

args = parser.parse_args()

file_path = args.file_path


lev_sum = 0.0
disc_sum = 0.0
count = 0

pattern = re.compile(r"Levenshtein_ratio=([0-9.]+), discard_ratio=([0-9.]+)")

with open(file_path, "r") as f:
    lines = f.readlines()[2:]  # 从第二行开始
    for line in lines:
        match = pattern.search(line)
        if match:
            lev = float(match.group(1))
            disc = float(match.group(2))
            lev_sum += lev
            disc_sum += disc
            count += 1

if count > 0:
    print("平均 Levenshtein_ratio:", lev_sum / count)
    print("平均 discard_ratio:", disc_sum / count)
else:
    print("没有匹配到数据")