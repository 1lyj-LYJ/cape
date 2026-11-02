# -*- coding: utf-8 -*-
"""Layout Completion, harder variant: hide TWO objects at once.

Same data, same models, same split as lc_task.py; only the task changes:
  * eligible layouts: >= 3 objects (>=1 visible object remains)
  * the two hidden objects are chosen by a fixed RNG seeded on the layout index
  * reference = the largest VISIBLE object
  * predict both centres (4 numbers) and both directions relative to the
    reference (two 4-class problems, sharing the same features)
  * features: the two target classes, visible class histogram, #visible,
    reference class+box, visible centre/area statistics. Nothing from the
    hidden boxes.
The point is to make the task sensitive to multi-object relative structure,
which is where CA-PE's advantage lives.
"""
import io, os, sys, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats as st

LC = os.path.join(ROOT, "runs", "lc")
vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
CIDX = {c: i for i, c in enumerate(vocab)}
NC = len(vocab)
DIM = NC * 4 + 1 + 4 + 4 + 2


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append((r.get("seed"), [{"cls": o["cls"], "box": o["box"]}
                                    for o in r["objects"]]))
    return out


def direction(cx, cy, rx, ry):
    dx, dy = cx - rx, cy - ry
    if abs(dx) >= abs(dy):
        return 0 if dx < 0 else 1
    return 2 if dy < 0 else 3


def featurize(objs, key):
    n = len(objs)
    if n < 3:
        return None
    area = [o["box"][2] * o["box"][3] for o in objs]
    rng = random.Random(key)
    hid = sorted(rng.sample(range(n), 2))
    vis = [i for i in range(n) if i not in hid]
    ref = max(vis, key=lambda i: area[i])
    f = np.zeros(DIM)
    f[CIDX[objs[hid[0]]["cls"]]] = 1.0
    f[NC + CIDX[objs[hid[1]]["cls"]]] = 1.0
    p = NC * 2
    for i in vis:
        f[p + CIDX[objs[i]["cls"]]] += 1.0
    f[p:p + NC] /= max(1, len(vis))
    p += NC
    f[p] = len(vis)
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
    Y, D = [], []
    rcx, rcy = rx + rw / 2, ry + rh / 2
    for j in hid:
        tx, ty, tw, th = objs[j]["box"]
        cx, cy = tx + tw / 2, ty + th / 2
        Y += [cx, cy]
        D.append(direction(cx, cy, rcx, rcy))
    return f, np.array(Y), np.array(D), hid


def build(rows, seeds=None):
    X, Y, D1, D2 = [], [], [], []
    for k, (s, objs) in enumerate(rows):
        if seeds is not None and s not in seeds:
            continue
        r = featurize(objs, f"{s}_{k}")
        if r is None:
            continue
        X.append(r[0])
        Y.append(r[1])
        D1.append(r[2][0])
        D2.append(r[2][1])
    return (np.array(X), np.array(Y), np.array(D1), np.array(D2))


def evaluate(Xtr, Ytr, A, B, Xte, Yte, Ate, Bte):
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, Ytr)
    P = ridge.predict(Xte)
    e1 = np.linalg.norm(P[:, :2] - Yte[:, :2], axis=1)
    e2 = np.linalg.norm(P[:, 2:] - Yte[:, 2:], axis=1)
    cerr = float(np.mean(np.concatenate([e1, e2])))
    accs, lls = [], []
    for dtr, dte in ((A, Ate), (B, Bte)):
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(C=1.0, max_iter=300)).fit(Xtr, dtr)
        pr = lr.predict_proba(Xte)
        accs.append(float((pr.argmax(1) == dte).mean()))
        lls.append(float(-np.mean(np.log(np.clip(pr[np.arange(len(dte)), dte],
                                                1e-12, 1.0)))))
    return cerr, float(np.mean(accs)), float(np.mean(lls))


def main():
    real_test = load(os.path.join(LC, "real_test.jsonl"))
    real_train = load(os.path.join(LC, "real_train.jsonl"))
    Xte, Yte, Ate, Bte = build(real_test)
    print(f"two-object task: {len(Xte)} layouts from the held-out 572 "
          f"(feature dim {Xte.shape[1]})")

    res = {}
    Xr, Yr, Ar, Br = build(real_train)
    prior = Yr.mean(axis=0, keepdims=True).repeat(len(Xte), axis=0)
    cerr = float(np.mean(np.concatenate([
        np.linalg.norm(prior[:, :2] - Yte[:, :2], axis=1),
        np.linalg.norm(prior[:, 2:] - Yte[:, 2:], axis=1)])))
    majs = [np.bincount(d).argmax() for d in (Ar, Br)]
    ps = [np.bincount(d).max() / len(d) for d in (Ar, Br)]
    accs, lls = [], []
    for maj, p_, dte in zip(majs, ps, (Ate, Bte)):
        accs.append(float((dte == maj).mean()))
        pm = np.full((len(dte), 4), (1 - p_) / 3)
        pm[:, maj] = p_
        lls.append(float(-np.mean(np.log(np.clip(pm[np.arange(len(dte)), dte],
                                                1e-12, 1.0)))))
    res["Simple prior"] = {"center_error": cerr, "dir_acc": float(np.mean(accs)),
                           "dir_logloss": float(np.mean(lls))}
    print(f"  Simple prior   centre {cerr:.4f}  dir {np.mean(accs):.4f}  "
          f"ll {np.mean(lls):.4f}")

    for arm in ("Publiconly", "ChamferCard", "LTSimCard", "CAPE"):
        rows = load(os.path.join(LC, f"{arm}.jsonl"))
        per = []
        for s in range(5):
            Xa, Ya, Aa, Ba = build(rows, seeds={s})
            r_ = evaluate(Xa, Ya, Aa, Ba, Xte, Yte, Ate, Bte)
            per.append(r_)
            print(f'    {arm} seed {s}: centre {r_[0]:.4f} dir {r_[1]:.4f}'
                  f' ll {r_[2]:.4f}', flush=True)
        res[arm] = {"center_error": float(np.mean([p[0] for p in per])),
                    "center_sd": float(np.std([p[0] for p in per], ddof=1)),
                    "dir_acc": float(np.mean([p[1] for p in per])),
                    "dir_acc_sd": float(np.std([p[1] for p in per], ddof=1)),
                    "dir_logloss": float(np.mean([p[2] for p in per])),
                    "dir_logloss_sd": float(np.std([p[2] for p in per], ddof=1)),
                    "per_seed": [{"center_error": p[0], "dir_acc": p[1],
                                  "dir_logloss": p[2]} for p in per]}
        r = res[arm]
        print(f"  {arm:14s} centre {r['center_error']:.4f}±{r['center_sd']:.4f}  "
              f"dir {r['dir_acc']:.4f}±{r['dir_acc_sd']:.4f}  "
              f"ll {r['dir_logloss']:.4f}±{r['dir_logloss_sd']:.4f}")

    cerr, acc, ll = evaluate(Xr, Yr, Ar, Br, Xte, Yte, Ate, Bte)
    res["Real-train"] = {"center_error": cerr, "dir_acc": acc, "dir_logloss": ll}
    print(f"  Real-train     centre {cerr:.4f}  dir {acc:.4f}  ll {ll:.4f}")

    print("\npaired vs CA-PE (5 seeds, two-object task)")
    for other in ("LTSimCard", "ChamferCard", "Publiconly"):
        for m, sgn in (("center_error", -1), ("dir_acc", 1), ("dir_logloss", -1)):
            a = np.array([p[m] for p in res["CAPE"]["per_seed"]])
            b = np.array([p[m] for p in res[other]["per_seed"]])
            t = st.ttest_rel(a, b)
            print(f"  {other:12s} {m:14s} CA {a.mean():.4f} vs {b.mean():.4f}"
                  f"  CA better {int(((a-b)*sgn>0).sum())}/5  p={t.pvalue:.4f}")

    print("\ngap closure (trivial prior -> real training)")
    f0, c0 = res["Simple prior"], res["Real-train"]
    for k in ("CAPE", "LTSimCard", "ChamferCard", "Publiconly"):
        row = f"  {k:14s}"
        for m, low in (("center_error", True), ("dir_acc", False),
                       ("dir_logloss", True)):
            room = abs(c0[m] - f0[m])
            gain = (f0[m] - res[k][m]) if low else (res[k][m] - f0[m])
            row += f"  {m}: {gain/room*100:6.0f}%"
        print(row)

    json.dump(res, open(os.path.join(LC, "completion_two_object.json"), "w"),
              indent=1)
    print("\nsaved runs/lc/completion_two_object.json")


if __name__ == "__main__":
    main()
