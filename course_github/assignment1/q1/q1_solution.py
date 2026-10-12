"""
Question 1: ERM / Naive Bayes / Fisher LDA classification on 4-D Gaussian data.
Generates 10K samples once, saves them, and reuses them for Parts A, B, C.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import eigh

rng = np.random.default_rng(0)
N = 10000

# ---------------- True distribution ----------------
P = np.array([0.35, 0.65])
m0 = np.full(4, -0.5)
C0 = 0.25 * np.array([[2, -0.5, 0.3, 0],
                      [-0.5, 1, -0.5, 0],
                      [0.3, -0.5, 1, 0],
                      [0, 0, 0, 2]])
m1 = np.ones(4)
C1 = np.array([[1, 0.3, -0.2, 0],
               [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0],
               [0, 0, 0, 3]])

# ---------------- Generate & save data ----------------
L = (rng.random(N) >= P[0]).astype(int)          # P(L=1)=0.65
X = np.zeros((N, 4))
n0, n1 = np.sum(L == 0), np.sum(L == 1)
X[L == 0] = rng.multivariate_normal(m0, C0, n0)
X[L == 1] = rng.multivariate_normal(m1, C1, n1)
np.savez("q1_data.npz", X=X, L=L)
print(f"Samples: N0={n0}, N1={n1}")


def roc_sweep(score, L):
    """Sweep threshold over all sorted scores (plus -inf/+inf).
    Decide D=1 when score > tau. Returns tau, FPR, TPR, P(error)."""
    s = np.sort(score)
    taus = np.concatenate(([-np.inf], (s[:-1] + s[1:]) / 2, [np.inf]))
    n0, n1 = np.sum(L == 0), np.sum(L == 1)
    s0, s1 = np.sort(score[L == 0]), np.sort(score[L == 1])
    # number of samples with score > tau
    fp = n0 - np.searchsorted(s0, taus, side="right")
    tp = n1 - np.searchsorted(s1, taus, side="right")
    fpr, tpr = fp / n0, tp / n1
    perr = fpr * (n0 / len(L)) + (1 - tpr) * (n1 / len(L))
    return taus, fpr, tpr, perr


def evaluate(score, name, theo_tau=None, log_scale=True):
    taus, fpr, tpr, perr = roc_sweep(score, L)
    i = np.argmin(perr)
    print(f"\n===== {name} =====")
    if log_scale:
        print(f"Empirical best gamma = {np.exp(taus[i]):.4f} (ln gamma = {taus[i]:.4f})")
    else:
        print(f"Empirical best tau = {taus[i]:.4f}")
    print(f"  min P(error) = {perr[i]:.4f}, TPR = {tpr[i]:.4f}, FPR = {fpr[i]:.4f}")
    out = dict(fpr=fpr, tpr=tpr, i=i, perr=perr[i])
    if theo_tau is not None:
        d = score > theo_tau
        tpr_t = np.mean(d[L == 1]); fpr_t = np.mean(d[L == 0])
        pe_t = fpr_t * n0 / N + (1 - tpr_t) * n1 / N
        print(f"Theoretical gamma = {np.exp(theo_tau):.4f}: P(error) = {pe_t:.4f}, "
              f"TPR = {tpr_t:.4f}, FPR = {fpr_t:.4f}")
        out.update(theo=(fpr_t, tpr_t, pe_t))
    return out


# 0-1 loss -> gamma* = P(L=0)/P(L=1)
gamma_theory = P[0] / P[1]
ln_gamma = np.log(gamma_theory)

# ---------------- Part A: true pdf ----------------
llr_A = mvn.logpdf(X, m1, C1) - mvn.logpdf(X, m0, C0)
resA = evaluate(llr_A, "Part A: ERM with true pdf", ln_gamma)

# ---------------- Part B: Naive Bayes (diagonal covariances) ----------------
C0nb, C1nb = np.diag(np.diag(C0)), np.diag(np.diag(C1))
llr_B = mvn.logpdf(X, m1, C1nb) - mvn.logpdf(X, m0, C0nb)
resB = evaluate(llr_B, "Part B: Naive Bayes", ln_gamma)

# ---------------- Part C: Fisher LDA ----------------
mu0_hat = X[L == 0].mean(axis=0)
mu1_hat = X[L == 1].mean(axis=0)
S0_hat = np.cov(X[L == 0].T)
S1_hat = np.cov(X[L == 1].T)
Sb = np.outer(mu1_hat - mu0_hat, mu1_hat - mu0_hat)   # between-class scatter
Sw = S0_hat + S1_hat                                   # within-class scatter (equal weights)
evals, evecs = eigh(Sb, Sw)                            # generalized eigenproblem Sb w = lambda Sw w
w = evecs[:, np.argmax(evals)]
if w @ (mu1_hat - mu0_hat) < 0:                        # orient so class 1 projects higher
    w = -w
print("\nEstimated means:\n mu0 =", mu0_hat.round(3), "\n mu1 =", mu1_hat.round(3))
print("w_LDA (unnormalised) =", w.round(4), " | unit:", (w / np.linalg.norm(w)).round(4))
proj_C = X @ w
resC = evaluate(proj_C, "Part C: Fisher LDA", log_scale=False)

# ---------------- Plots ----------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, res, title in zip(axes, [resA, resB, resC],
                          ["Part A: ERM (true pdf)", "Part B: Naive Bayes", "Part C: Fisher LDA"]):
    ax.plot(res["fpr"], res["tpr"], lw=1.5, label="ROC")
    ax.plot(res["fpr"][res["i"]], res["tpr"][res["i"]], "ro", ms=9,
            label=f"Empirical min P(err)={res['perr']:.4f}")
    if "theo" in res:
        ax.plot(res["theo"][0], res["theo"][1], "gx", ms=11, mew=2,
                label=f"Theoretical gamma={gamma_theory:.3f}, P(err)={res['theo'][2]:.4f}")
    ax.set(xlabel="P(D=1|L=0)  (False positive)", ylabel="P(D=1|L=1)  (True positive)",
           title=title, xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("q1_roc_curves.png", dpi=150)

plt.figure(figsize=(6, 6))
for res, lab in zip([resA, resB, resC], ["ERM (true pdf)", "Naive Bayes", "Fisher LDA"]):
    plt.plot(res["fpr"], res["tpr"], label=f"{lab} (min P(err)={res['perr']:.4f})")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC comparison"); plt.grid(alpha=0.3); plt.legend(loc="lower right")
plt.xlim(0, 0.4); plt.ylim(0.6, 1.0)
plt.tight_layout(); plt.savefig("q1_roc_comparison_zoom.png", dpi=150)
