# -*- coding: utf-8 -*-
"""Layout Completion (TSTR): train on synthetic layouts, test on held-out real.

PRE-DECLARED PROTOCOL
  task     hide one object's box, keep its class; predict its centre and its
           direction relative to a reference object
  split    2287 real layouts for synthesising the DP training data,
           572 real layouts held out for testing (disjoint image_ids)
  target   every object j of a layout with >=2 objects; reference = the largest
           VISIBLE object (never j)
  features target class one-hot, visible class histogram, #visible, reference
           class one-hot + box, mean/std of visible centres, mean/std of visible
           areas.  NOTHING derived from the hidden box enters the features.
  models   Ridge (centre, alpha=1.0) and multinomial Logistic (direction, C=1.0)
           -- identical for every training source
  metrics  Centre Error (normalised L2), Direction Accuracy, Direction log-loss
  sources  Simple-prior (train-set mean centre, most frequent direction),
           Public-only, Chamfer+Card, LTSim+Card, CA-PE, Real-train (non-DP)
  unit     5 outer DP seeds; each seed trains on that seed's own release
"""
import io, os, sys, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from layout.representation import Layout

LC = os.path.join(ROOT, "runs", "lc")
vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
CIDX = {c: i for i, c in enumerate(vocab)}
NC = len(vocab)
DIRS = ["left", "right", "above", "below"]


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append((r.get("seed"), [{"cls": o["cls"], "box": o["box"]}
                                    for o in r["objects"]]))
    return out


def featurize(objs):
    """return X rows, centres y, directions d, for every hideable target"""
    n = len(objs)
    if n < 2:
        return [], [], []
    area = [o["box"][2] * o["box"][3] for o in objs]
    X, Y, D = [], [], []
    for j in range(n):
        vis = [i for i in range(n) if i != j]
        if not vis:
            continue
        ref = max(vis, key=lambda i: area[i])
        f = np.zeros(NC * 3 + 1 + 4 + 4 + 2)
        f[CIDX[objs[j]["cls"]]] = 1.0
        base = NC
        cnt = 0
        for i in vis:
            f[base + CIDX[objs[i]["cls"]]] += 1.0
            cnt += 1
        if cnt:
            f[base:base + NC] /= cnt
        p = NC * 2
        f[p] = cnt
        p += 1
        f[p + CIDX[objs[ref]["cls"]]] = 1.0
        p += NC
        rx, ry, rw, rh = objs[ref]["box"]
        f[p:p + 4] = [rx + rw / 2, ry + rh / 2, rw, rh]
        p += 4
        cs = np.array([[objs[i]["box"][0] + objs[i]["box"][2] / 2,
                        objs[i]["box"][1] + objs[i]["box"][3] / 2] for i in vis])
        f[p:p + 2] = cs.mean(axis=0)
        f[p + 2:p + 4] = cs.std(axis=0)
        p += 4
        ar = np.array([area[i] for i in vis])
        f[p] = ar.mean()
        f[p + 1] = ar.std()
        X.append(f)
        tx, ty, tw, th = objs[j]["box"]
        cx, cy = tx + tw / 2, ty + th / 2
        Y.append([cx, cy])
        dx, dy = cx - (rx + rw / 2), cy - (ry + rh / 2)
        if abs(dx) >= abs(dy):
            D.append(0 if dx < 0 else 1)
        else:
            D.append(2 if dy < 0 else 3)
    return X, Y, D


def build(rows, seeds=None):
    X, Y, D = [], [], []
    for s, objs in rows:
        if seeds is not None and s not in seeds:
            continue
        x, y, d = featurize(objs)
        X += x
        Y += y
        D += d
    return np.array(X), np.array(Y), np.array(D)


def evaluate(name, Xtr, Ytr, Dtr, Xte, Yte, Dte, out):
    ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)
    P = ridge.predict(Xte)
    cerr = float(np.mean(np.linalg.norm(P - Yte, axis=1)))
    if len(set(Dtr)) >= 2:
        logi = LogisticRegression(C=1.0, max_iter=2000)
        logi.fit(Xtr, Dtr)
        prob = logi.predict_proba(Xte)
        acc = float((prob.argmax(1) == Dte).mean())
        eps = 1e-12
        ll = float(-np.mean(np.log(np.clip(
            prob[np.arange(len(Dte)), Dte], eps, 1.0))))
    else:
        acc, ll = float("nan"), float("nan")
    out[name] = {"center_error": cerr, "dir_acc": acc, "dir_logloss": ll,
                 "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    print(f"  {name:16s} centre {cerr:.4f}  dir-acc {acc:.4f}  "
          f"logloss {ll:.4f}   (train {len(Xtr)}, test {len(Xte)})")


def main():
    real_test = load(os.path.join(LC, "real_test.jsonl"))
    real_train = load(os.path.join(LC, "real_train.jsonl"))
    Xte, Yte, Dte = build(real_test)
    print(f"test set: {len(Xte)} completion instances from "
          f"{len(real_test)} real layouts (feature dim {Xte.shape[1]})")

    res = {}
    Xr, Yr, Dr = build(real_train)
    # simple prior: training-set means only
    prior_c = Yr.mean(axis=0, keepdims=True).repeat(len(Xte), axis=0)
    cerr = float(np.mean(np.linalg.norm(prior_c - Yte, axis=1)))
    vals, cnts = np.unique(Dr, return_counts=True)
    maj = int(vals[cnts.argmax()])
    acc = float((Dte == maj).mean())
    p_maj = float(cnts.max() / cnts.sum())
    uni = np.full((len(Dte), 4), 1.0 / 4)
    pm = np.full((len(Dte), 4), (1 - p_maj) / 3)
    pm[:, maj] = p_maj
    ll = float(-np.mean(np.log(np.clip(pm[np.arange(len(Dte)), Dte], 1e-12, 1))))
    res["Simple prior"] = {"center_error": cerr, "dir_acc": acc,
                           "dir_logloss": ll, "n_train": 0, "n_test": len(Xte)}
    print(f"  Simple prior     centre {cerr:.4f}  dir-acc {acc:.4f}  logloss {ll:.4f}")

    for arm in ("Publiconly", "ChamferCard", "LTSimCard", "CAPE"):
        rows = load(os.path.join(LC, f"{arm}.jsonl"))
        per = []
        for s in range(5):
            Xtr, Ytr, Dtr = build(rows, seeds={s})
            tmp = {}
            evaluate(f"{arm}_s{s}", Xtr, Ytr, Dtr, Xte, Yte, Dte, tmp)
            per.append(tmp[f"{arm}_s{s}"])
        res[arm] = {"per_seed": per,
                    "center_error": float(np.mean([p["center_error"] for p in per])),
                    "center_sd": float(np.std([p["center_error"] for p in per], ddof=1)),
                    "dir_acc": float(np.mean([p["dir_acc"] for p in per])),
                    "dir_acc_sd": float(np.std([p["dir_acc"] for p in per], ddof=1)),
                    "dir_logloss": float(np.mean([p["dir_logloss"] for p in per])),
                    "dir_logloss_sd": float(np.std([p["dir_logloss"] for p in per], ddof=1)),
                    "n_train": per[0]["n_train"], "n_test": len(Xte)}

    evaluate("Real-train", Xr, Yr, Dr, Xte, Yte, Dte, res)

    print("\nSUMMARY (mean over 5 outer seeds)")
    print(f"{'source':16s}{'centre ↓':>12}{'dir-acc ↑':>12}{'logloss ↓':>12}")
    for k, v in res.items():
        if "per_seed" in v:
            print(f"{k:16s}{v['center_error']:>8.4f}±{v['center_sd']:.3f}"
                  f"{v['dir_acc']:>8.4f}±{v['dir_acc_sd']:.3f}"
                  f"{v['dir_logloss']:>8.4f}±{v['dir_logloss_sd']:.3f}")
        else:
            print(f"{k:16s}{v['center_error']:>12.4f}{v['dir_acc']:>12.4f}"
                  f"{v['dir_logloss']:>12.4f}")

    # paired tests CA-PE vs the others (5 seeds each)
    if "CAPE" in res:
        from scipy import stats as st
        print("\npaired vs CA-PE (5 seeds)")
        for other in ("LTSimCard", "ChamferCard", "Publiconly"):
            for m, sgn in (("center_error", -1), ("dir_acc", 1), ("dir_logloss", -1)):
                a = np.array([p[m] for p in res["CAPE"]["per_seed"]])
                b = np.array([p[m] for p in res[other]["per_seed"]])
                d = (a - b) * sgn            # >0 means CA-PE better
                t = st.ttest_rel(a, b)
                print(f"  {other:12s} {m:14s} CA {a.mean():.4f} vs {b.mean():.4f}"
                      f"  (CA-PE better {int((d>0).sum())}/5, p={t.pvalue:.4f})")
    json.dump(res, open(os.path.join(LC, "completion_results.json"), "w"), indent=1)
    print("\nsaved runs/lc/completion_results.json")


if __name__ == "__main__":
    main()
