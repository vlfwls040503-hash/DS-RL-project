# -*- coding: utf-8 -*-
"""
27_raw_cnn_eval.py  --  D1: 원시궤적 CNN 평가자 (수제 특징을 걷어낸 최종 시험대).

수제 8특징 C2ST는 우리가 '보기로 한 것'만 본다. 여기서는 원시 다채널 시계열
[e, psi, 횡속, 횡가속, theta] (10m 그리드) 창을 1D CNN 판별자에 통째로 넣어
챔피언 v3.1의 진짜 위장 실력을 측정한다.

누수 방지: 유닛(사람 5.2km 청크 / 챔피언 롤아웃) 단위 GroupKFold — 같은 유닛의
창이 학습/시험에 갈라지지 않음. 셔플라벨 대조(유닛 단위 셔플)로 ~0.5 확인.

  python 27_raw_cnn_eval.py --per_road 10
"""
import os, json, argparse, importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p24 = importlib.import_module("24_gail_seg")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS = ["e", "psi", "latv", "lata", "theta"]


def unit_windows(sig, w, stride):
    """유닛 신호 → (n, 5, w) 원시 창 배열."""
    n = len(sig["e"])
    out = []
    for a in range(0, n - w + 1, stride):
        out.append(np.stack([np.asarray(sig[c][a:a + w], "float32") for c in CHANNELS]))
    return out


class RawCNN(nn.Module):
    def __init__(self, ch=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2, stride=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.conv(x)
        z = torch.cat([h.mean(-1), h.max(-1).values], dim=1)
        return self.head(z).squeeze(-1)


def cnn_group_cv(X, y, groups, seed=0, epochs=30, batch=256):
    """유닛 GroupKFold 5-fold: 채널 정규화는 train fold 통계로. 풀링된 AUC."""
    gkf = GroupKFold(5)
    ps = np.zeros(len(y))
    for fold, (tri, tei) in enumerate(gkf.split(X, y, groups)):
        torch.manual_seed(seed * 100 + fold)
        mu = X[tri].mean(axis=(0, 2), keepdims=True)
        sd = X[tri].std(axis=(0, 2), keepdims=True) + 1e-8
        Xtr = torch.from_numpy((X[tri] - mu) / sd).to(DEV)
        ytr = torch.from_numpy(y[tri].astype("float32")).to(DEV)
        Xte = torch.from_numpy((X[tei] - mu) / sd).to(DEV)
        net = RawCNN(X.shape[1]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            perm = torch.randperm(len(Xtr), device=DEV)
            for s in range(0, len(Xtr), batch):
                b = perm[s:s + batch]
                opt.zero_grad()
                loss = bce(net(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        with torch.no_grad():
            ps[tei] = torch.sigmoid(net(Xte)).cpu().numpy()
    return float(roc_auc_score(y, ps))


def feat_group_cv(X, y, groups, seed=0):
    """동일 창에서 수제 8특징 로지스틱 — 사과대사과 기준선."""
    F = np.stack([p24.window_feats(x[0], x[2], x[3], x[4]) for x in X])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    gkf = GroupKFold(5)
    ps = np.zeros(len(y))
    for tri, tei in gkf.split(F, y, groups):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(F[tri], y[tri]); ps[tei] = clf.predict_proba(F[tei])[:, 1]
    return float(roc_auc_score(y, ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_road", type=int, default=10)
    args = ap.parse_args()

    # ---- 챔피언 v3.1 자산 + 2024 test 프로토콜 (22와 동일) ----
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    fit_roads = [r for r, m in zip(roads, tr | va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]
    fit_chunks = []
    for r in fit_roads:
        for ch in p21.chunk_signals(p20.human_signals(r, dd)):
            fit_chunks.append(ch["e"])
    fr, A = p22.target_spectrum(fit_chunks)
    lib = []
    for e in fit_chunks:
        x = np.asarray(e, np.float64); x = x - x.mean()
        if len(x) >= 400 and x.std() > 1e-3:
            lib.append((x / x.std()).astype(np.float64))
    model = PPO.load(os.path.join(ART, "rl_2024.zip"), device="cpu")
    cal = json.load(open(os.path.join(REP, "v3_library_2024.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"])), cond=int(r.get("cond", 0))) for r in fit_roads]
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))
    by_cond = {}
    for t in pool:
        by_cond.setdefault(t["cond"], []).append(t)

    # ---- 유닛 수집: 사람 청크 vs 챔피언 롤아웃 ----
    H_units = []
    for road in test_roads:
        H_units.extend(p21.chunk_signals(p20.human_signals(road, dd)))
    env = DrivingEnv(test_roads, dd=dd, record=True)
    rng = np.random.RandomState(0)
    S_units = []
    for k, road in enumerate(test_roads):
        cand = by_cond.get(int(road.get("cond", 0)), pool)
        for j in range(args.per_road):
            t = cand[rng.randint(len(cand))]
            pol = p22.SpectralPolicy(model, fr, A,
                                     sigma=float(np.clip(sigma * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                                     b_bias=t["lpm"], v_scale=t["v"] / v_pool, lib=lib,
                                     seed=9000 + k * 20 + j)
            pol.reset()
            traj, _ = rollout(env, pol, k)
            if len(traj) > 60:
                S_units.append(p20.rl_signals(traj))
        if (k + 1) % 5 == 0:
            print(f"  rollout road {k+1}/{len(test_roads)}", flush=True)
    nu = min(len(H_units), len(S_units))
    rngb = np.random.RandomState(1)
    H_units = [H_units[i] for i in rngb.choice(len(H_units), nu, replace=False)]
    S_units = [S_units[i] for i in rngb.choice(len(S_units), nu, replace=False)]
    print(f"units: human {nu} / champion {nu} (밸런스드)", flush=True)

    # ---- 창 추출 → CNN / 특징 로지스틱 / 셔플 대조 ----
    results = {}
    for w, tag in [(20, "200m"), (100, "1000m")]:
        Xs, ys, gs = [], [], []
        gid = 0
        for cls, units in [(0, H_units), (1, S_units)]:
            for u in units:
                for win in unit_windows(u, w, w // 2):
                    Xs.append(win); ys.append(cls); gs.append(gid)
                gid += 1
        X = np.stack(Xs); y = np.array(ys); g = np.array(gs)
        auc_cnn = cnn_group_cv(X, y, g)
        entry = dict(n_win=int(len(y)), auc_cnn=auc_cnn)
        if w == 20:
            entry["auc_feat"] = feat_group_cv(X, y, g)
            rs = np.random.RandomState(7)                 # 유닛 단위 라벨 셔플 (누수 검사)
            uids = np.unique(g)
            umap = dict(zip(uids, rs.permutation([int(y[g == u][0]) for u in uids])))
            y_sh = np.array([umap[gi] for gi in g])
            entry["auc_shuffle"] = cnn_group_cv(X, y_sh, g, seed=7)
        results[tag] = entry
        print(f"[{tag}] n_win={len(y)} CNN AUC={auc_cnn:.3f}"
              + (f" | feat8={entry['auc_feat']:.3f} | shuffle={entry['auc_shuffle']:.3f}"
                 if w == 20 else ""), flush=True)

    # ---- 그림 + 저장 ----
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = ["특징8종\n(200m창)", "CNN 원시\n(200m창)", "CNN 원시\n(1000m창)", "셔플 대조\n(누수검사)"]
    vals = [results["200m"]["auc_feat"], results["200m"]["auc_cnn"],
            results["1000m"]["auc_cnn"], results["200m"]["auc_shuffle"]]
    ax.bar(names, vals, color=["#888780", "#7F77DD", "#5A54A8", "#1D9E75"])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("GroupKFold AUC")
    ax.set_title("원시궤적 CNN 평가자: 챔피언 v3.1의 진짜 위장 실력")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rawcnn_2024.png"), dpi=120)
    plt.close(fig)
    json.dump(dict(exp="2024", device=DEV, n_units=int(nu), results=results),
              open(os.path.join(REP, "rawcnn_2024.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fig_rawcnn_2024.png + rawcnn_2024.json", flush=True)


if __name__ == "__main__":
    main()
