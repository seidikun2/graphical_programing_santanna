from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = Path("/home/seidi/Área de trabalho")
f1 = FOLDER / "test_signal1.csv"
f2 = FOLDER / "test_signal2.csv"

def read_lv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, sep="\t", decimal=",")

df1 = read_lv(f1)  # tempo + sinais
df2 = read_lv(f2)

t1 = df1.iloc[:, 0].to_numpy()
t2 = df2.iloc[:, 0].to_numpy()

# sinais = colunas após a 1ª
X1 = df1.iloc[:, 1:]
X2 = df2.iloc[:, 1:]

# --------- Figura com 2 painéis: (1) sinais, (2) t1 vs t2 ----------
fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.25)

ax_sig = fig.add_subplot(gs[0])
ax_t   = fig.add_subplot(gs[1])

# Paletas em "tons" diferentes por arquivo
cmap1 = plt.cm.Blues
cmap2 = plt.cm.Oranges

# gera tons semelhantes dentro do mesmo arquivo
def tones(cmap, n, lo=0.45, hi=0.95):
    if n <= 1:
        return [cmap(hi)]
    return [cmap(v) for v in np.linspace(lo, hi, n)]

cols1 = tones(cmap1, X1.shape[1])
cols2 = tones(cmap2, X2.shape[1])

# --- Painel 1: sinais sobrepostos ---
for j in range(X1.shape[1]):
    ax_sig.plot(t1, X1.iloc[:, j].to_numpy(),
                color=cols1[j], linewidth=2,
                label=f"sig1 col{j+2}")  # +2 porque col 1 é tempo, sinais começam na col 2 (contando de 1)

for j in range(X2.shape[1]):
    ax_sig.plot(t2, X2.iloc[:, j].to_numpy(),
                color=cols2[j], linewidth=2,
                label=f"sig2 col{j+2}")

ax_sig.set_title("Sinais sobrepostos (tons por arquivo)")
ax_sig.set_xlabel("Tempo")
ax_sig.set_ylabel("Amplitude")
ax_sig.grid(True)
ax_sig.legend(ncols=3, fontsize=9)

# --- Painel 2: tempo de um pelo outro (t1 vs t2) ---
n = min(len(t1), len(t2))
t1n, t2n = t1[:n], t2[:n]

ax_t.plot(t1n, t2n, marker="o", linestyle="", markersize=4, alpha=0.8, label="t2 vs t1")

# linha y=x para referência (mesma escala)
mn = np.nanmin([np.nanmin(t1n), np.nanmin(t2n)])
mx = np.nanmax([np.nanmax(t1n), np.nanmax(t2n)])
ax_t.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.5, label="y = x")

ax_t.set_title("Tempo de um pelo outro (sincronia)")
ax_t.set_xlabel("t1 (arquivo 1)")
ax_t.set_ylabel("t2 (arquivo 2)")
ax_t.grid(True)
ax_t.legend(fontsize=9)

plt.tight_layout()
plt.show()