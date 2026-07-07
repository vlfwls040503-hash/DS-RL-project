# -*- coding: utf-8 -*-
"""
54_fewshot_namsan.py  --  남산 퓨샷 프로토콜: train-피험자 질감 투입, test-피험자 시험.

파일럿 보정(§14 퓨샷)의 확장판 — 텍스처 자체를 파일럿(19명)에서 학습. 누수 없음:
시험은 피험자-분리 test(10명)만. 기준선(현직 46d+42, 52스택)도 동일 te-전용 시험으로
재측정해 공정 비교. "퓨샷" 라벨 명시.

  python 54_fewshot_namsan.py
"""
import os, json, copy, time, importlib
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, OBS_DIM

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")
p44 = importlib.import_module("44_bc_native")
p51 = importlib.import_module("51_trackA_diffusion")
p52 = importlib.import_module("52_best_stack")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012
CKPT_FS = os.path.join(ART, "trackA54_fewshot.pt")


def train_fewshot(train_roads, dd):
    C, A = p51.make_sequences(train_roads, dd)
    print(f"fewshot sequences: {len(C):,} | resid std={A.std():.3f}", flush=True)
    mu_c, sd_c = C.mean(0), C.std(0) + 1e-6
    Cn = (C - mu_c) / sd_c
    r_sd = float(A.std()) + 1e-6
    An = A / r_sd
    net = p51.Denoiser().to(p51.DEV)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    ab = p51.cosine_schedule(p51.T_DIFF).to(p51.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    Ct = torch.from_numpy(Cn); At = torch.from_numpy(An)
    for ep in range(80):
        perm = torch.randperm(len(Ct))
        for s in range(0, len(Ct), 1024):
            b = perm[s:s + 1024]
            x0 = At[b].to(p51.DEV); cond = Ct[b].to(p51.DEV)
            t = torch.randint(0, p51.T_DIFF, (len(b),), device=p51.DEV)
            eps = torch.randn_like(x0)
            a = ab[t].view(-1, 1, 1)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            opt.zero_grad()
            loss = ((net(xt, t, cond) - eps) ** 2).mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.999).add_(pn, alpha=0.001)
    torch.save(dict(state=ema.state_dict(), mu_c=mu_c, sd_c=sd_c, r_sd=r_sd), CKPT_FS)


def main():
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)

    # ---- 남산 분할 ----
    rN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    rN = trim_roads(rN)
    sN = np.array([r["subject"] for r in rN], "int64")
    trN, vaN, teN = gen_split(sN, seed=0)
    tr_r = [r for r, m in zip(rN, trN) if m]
    va_r = [r for r, m in zip(rN, vaN) if m]
    te_r = [r for r, m in zip(rN, teN) if m]
    print(f"namsan tr/va/te roads: {len(tr_r)}/{len(va_r)}/{len(te_r)}", flush=True)

    # ---- 퓨샷 코퍼스: 청정 저속 + 남산 train ----
    m8, _, dd8 = load_roads(os.path.join(CACHE, "env_roads_multi8.npz"))
    m8 = trim_roads(m8)
    s8 = np.array([r["subject"] for r in m8], "int64")
    t8, v8, _ = gen_split(s8, seed=0)
    corpus = [r for r, m in zip(m8, t8 | v8) if m]
    for extra in ["icing", "underpass21"]:
        p_ = os.path.join(CACHE, f"env_roads_{extra}.npz")
        ex, _, _d = load_roads(p_)
        corpus += trim_roads(ex)
    corpus += tr_r                                   # ★ 남산 파일럿 19명
    train_fewshot(corpus, dd8)

    # ---- 남산 청크 라이브러리: train 피험자 (길이 완화 >=200) ----
    libN = []
    for r in tr_r:
        x = np.asarray(p20.human_signals(r, ddN)["e"], np.float64)
        x -= x.mean()
        if len(x) >= 200 and x.std() > 1e-3:
            libN.append((x / x.std()).astype(np.float64))
    print(f"namsan pilot chunk lib: {len(libN)}", flush=True)

    # 퓨샷 정책: 52 스택 + 퓨샷 체크포인트 + 남산 청크
    class FewshotPolicy(p52.BestStackPolicy):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            d = torch.load(CKPT_FS, map_location=p51.DEV, weights_only=False)
            self.net = p51.Denoiser().to(p51.DEV)
            self.net.load_state_dict(d["state"]); self.net.eval()
            self.mu, self.sd, self.r_sd = d["mu_c"], d["sd_c"], float(d["r_sd"])

    env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in va_r],
                       dd=ddN, record=True, steer_gain=GAIN)
    hv = [p20.human_signals(r, ddN) for r in va_r]
    tgt_sd = float(np.mean([np.std(h["e"]) for h in hv]))
    tgt_de = float(np.mean([np.mean(np.abs(np.diff(h["e"]))) for h in hv]))
    best = None
    for beta in [1.0, 2.0]:
        for tau in [0.6, 1.0, 1.5]:
            s0 = 0.2
            for it in range(2):
                sds, des = [], []
                for k in range(len(va_r)):
                    pol = FewshotPolicy(bc, beta=beta, tau=tau, sigma=s0, lib=libN,
                                        seed=90 + k)
                    pol.reset()
                    traj, _ = rollout(env_v, pol, k)
                    if len(traj) > 60:
                        sg = p39.symmetric_signals(traj, va_r[k], ddN, GAIN)
                        sds.append(float(np.std(sg["e"])))
                        des.append(float(np.mean(np.abs(np.diff(sg["e"])))))
                s0 = float(np.clip(s0 * tgt_sd / max(np.mean(sds), 1e-6), 0.03, 1.0))
            gap = abs(np.mean(des) - tgt_de) / tgt_de \
                + 0.5 * abs(np.mean(sds) - tgt_sd) / tgt_sd
            print(f"  b={beta} tau={tau} σ={s0:.3f}: |de|={np.mean(des):.4f}({tgt_de:.4f})"
                  f" SDLP={np.mean(sds):.3f}({tgt_sd:.3f}) gap={gap:.3f}", flush=True)
            if best is None or gap < best[3]:
                best = (beta, tau, s0, gap)
    beta_b, tau_b, sig_b, _ = best
    print(f"선택 b={beta_b} tau={tau_b} σ={sig_b:.3f}", flush=True)

    # ---- te-전용 시험: 퓨샷 vs 기준선(현직 46d+42 구성) ----
    Hte = [p21.seg_features(p20.human_signals(r, ddN)) for r in te_r]
    blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in te_r]
    env = DrivingEnv(blind, dd=ddN, record=True, steer_gain=GAIN)

    def exam(mk_pol, tag):
        T, offc = [], 0
        for k in range(len(blind)):
            for j in range(4):
                pol = mk_pol(7000 + k * 20 + j)
                pol.reset()
                traj, off = rollout(env, pol, k)
                offc += int(off)
                if len(traj) > 60:
                    T.append((traj, te_r[k]))
        S = [p21.seg_features(p39.symmetric_signals(t, r, ddN, GAIN)) for t, r in T]
        auc = p43.fair_exam(list(Hte), S)
        print(f"[te-전용] {tag}: AUC={auc:.3f} off={offc/max(len(blind)*4,1):.2f}",
              flush=True)
        return float(auc)

    # 기준선: 2024 청크 + β2 (46d+42 구성 근사, 기존 잔차 ckpt)
    r24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    r24 = trim_roads(r24)
    s24 = np.array([r["subject"] for r in r24], "int64")
    _, _, te24 = gen_split(s24, seed=0)
    fitlib = []
    for r in [r for r, m in zip(r24, ~te24) if m]:
        for ch_ in p21.chunk_signals(p20.human_signals(r, dd24)):
            x = np.asarray(ch_["e"], np.float64); x -= x.mean()
            if len(x) >= 400 and x.std() > 1e-3:
                fitlib.append((x / x.std()).astype(np.float64))
    base_auc = exam(lambda sd_: p52.BestStackPolicy(bc, beta=2.0, tau=0.0, sigma=0.142,
                                                    lib=fitlib, seed=sd_), "기준선(현직)")
    fs_auc = exam(lambda sd_: FewshotPolicy(bc, beta=beta_b, tau=tau_b, sigma=sig_b,
                                            lib=libN, seed=sd_), "퓨샷")
    json.dump(dict(baseline_te=base_auc, fewshot_te=fs_auc,
                   knobs=dict(beta=beta_b, tau=tau_b, sigma=sig_b),
                   n_pilot=len(tr_r), n_test=len(te_r)),
              open(os.path.join(REP, "fewshot_namsan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fewshot_namsan.json", flush=True)


if __name__ == "__main__":
    main()
