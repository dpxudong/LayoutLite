import os
import torch
from tqdm import tqdm

alpha_dict_firered = {
    0: 10,
    0.05: 0.110081,
    0.1: 0.086760,
    0.15: 0.070260,
    0.2: 0.056182,
    0.25: 0.042790,
    0.3: 0.028937,
    0.35: 0.013058,
    0.4: -0.007598,
    0.45: -0.040915,
    0.5: -0.106547,
    0.55: -0.283551,
    0.6: -0.683692,
    0.65: -0.913936,
    0.7: -0.985278,
    0.75: -1.014321,
    0.8: -1.035081
}
alpha_dict_logics = {
    0: 10,
    0.05: 0.111035,
    0.1: 0.088979,
    0.15: 0.073862,
    0.2: 0.061206,
    0.25: 0.049205,
    0.3: 0.036986,
    0.35: 0.022397,
    0.4: 0.003260,
    0.45: -0.028101,
    0.5: -0.091795,
    0.55: -0.232760,
    0.6: -0.555910,
    0.65: -0.942688,
    0.7: -1.005481,
    0.75: -1.019032,
    0.8: -1.028868
}

def find_alpha(input_dir, target):
    
    INPUT_DIR = input_dir

    TARGETS = [target]


    def get_centers(tensor):
        flat = tensor.flatten().float()

        c0 = flat.min()
        c1 = flat.max()

        if c0 == c1:
            return None

        for _ in range(5):
            labels = (flat - c0).abs() > (flat - c1).abs()

            if (~labels).any():
                c0 = flat[~labels].mean()

            if labels.any():
                c1 = flat[labels].mean()

        if c0 > c1:
            c0, c1 = c1, c0

        return float(c0), float(c1)


    # 预读取所有数据
    all_data = []

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".pt")
    )

    for fname in tqdm(files):
        tensor = torch.load(
            os.path.join(INPUT_DIR, fname),
            map_location="cpu"
        )

        centers = get_centers(tensor)

        if centers is None:
            continue

        c0, c1 = centers

        all_data.append(
            {
                "data": tensor.flatten().float(),
                "c0": c0,
                "c1": c1,
            }
        )

    print(f"loaded {len(all_data)} pt files")


    def mean_small_ratio(alpha):
        ratios = []

        for item in all_data:
            threshold = (
                item["c0"]
                - alpha * (item["c1"] - item["c0"])
            )

            ratio = (
                item["data"] <= threshold
            ).float().mean().item()

            ratios.append(ratio)

        return sum(ratios) / len(ratios)


    def search_alpha(target_ratio,
                    left=-10,
                    right=10,
                    max_iter=50):

        for _ in range(max_iter):
            mid = (left + right) / 2

            ratio = mean_small_ratio(mid)

            if ratio > target_ratio:
                # 小类太多
                left = mid
            else:
                right = mid

        alpha = (left + right) / 2

        return alpha, mean_small_ratio(alpha)


    for target in TARGETS:
        alpha, real_ratio = search_alpha(target)

    return alpha