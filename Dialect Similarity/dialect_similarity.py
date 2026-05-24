"""
Arabic Dialect Similarity Analysis
====================================
Compares Arabic dialects using three complementary methods:
  1. Character n-gram TF-IDF + Cosine Similarity
  2. Word-level Jaccard Similarity
  3. Levenshtein (edit distance) Similarity

For each dialect, produces a ranking of all other dialects
from most to least similar.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# ─────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/MADAR_combined_Arabic_2_.csv')
dialects = [c for c in df.columns if c != 'ID']
print(f"Loaded {len(df):,} sentences × {len(dialects)} dialects")
print(f"Dialects: {dialects}\n")

# ─────────────────────────────────────────────
# 2.  Helper: join all sentences for one dialect
# ─────────────────────────────────────────────
def corpus(dialect):
    """Return list of sentences for a dialect (drop NaNs)."""
    return df[dialect].dropna().tolist()

# ─────────────────────────────────────────────
# 3.  Method A – Character n-gram TF-IDF Cosine
# ─────────────────────────────────────────────
def char_ngram_similarity(dialects, df):
    """
    For each dialect, concatenate all its sentences into one document.
    Fit a character-level TF-IDF (2-4 grams) and compute pairwise cosine sim.
    """
    docs = [" ".join(df[d].dropna().tolist()) for d in dialects]
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X)
    return pd.DataFrame(sim, index=dialects, columns=dialects)

# ─────────────────────────────────────────────
# 4.  Method B – Word-level Jaccard Similarity
# ─────────────────────────────────────────────
def jaccard_similarity(dialects, df):
    """
    For each dialect build a vocabulary (set of unique words).
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    vocabs = {}
    for d in dialects:
        words = set()
        for sent in df[d].dropna():
            words.update(str(sent).split())
        vocabs[d] = words

    n = len(dialects)
    sim = np.zeros((n, n))
    for i, a in enumerate(dialects):
        for j, b in enumerate(dialects):
            inter = len(vocabs[a] & vocabs[b])
            union = len(vocabs[a] | vocabs[b])
            sim[i, j] = inter / union if union else 0.0

    return pd.DataFrame(sim, index=dialects, columns=dialects)

# ─────────────────────────────────────────────
# 5.  Method C – Sentence-level word overlap
#     (avg per-sentence Jaccard across shared IDs)
# ─────────────────────────────────────────────
def sentence_overlap_similarity(dialects, df):
    """
    For every sentence pair (same row), compute word-overlap ratio,
    then average across all sentences.
    """
    n = len(dialects)
    sim = np.zeros((n, n))

    for i, a in enumerate(dialects):
        for j, b in enumerate(dialects):
            if i == j:
                sim[i, j] = 1.0
                continue
            scores = []
            for sa, sb in zip(df[a].dropna(), df[b].dropna()):
                wa = set(str(sa).split())
                wb = set(str(sb).split())
                union = wa | wb
                if union:
                    scores.append(len(wa & wb) / len(union))
            sim[i, j] = np.mean(scores) if scores else 0.0

    return pd.DataFrame(sim, index=dialects, columns=dialects)

# ─────────────────────────────────────────────
# 6.  Compute all three matrices
# ─────────────────────────────────────────────
print("Computing character n-gram TF-IDF cosine similarity …")
sim_char = char_ngram_similarity(dialects, df)

print("Computing vocabulary-level Jaccard similarity …")
sim_jaccard = jaccard_similarity(dialects, df)

print("Computing sentence-level word-overlap similarity …")
sim_sentence = sentence_overlap_similarity(dialects, df)

# ─────────────────────────────────────────────
# 7.  Ensemble: simple average of all three
# ─────────────────────────────────────────────
sim_ensemble = (sim_char + sim_jaccard + sim_sentence) / 3.0
print("Ensemble similarity matrix computed.\n")

# ─────────────────────────────────────────────
# 8.  Rankings per dialect
# ─────────────────────────────────────────────
def build_ranking_table(sim_matrix):
    rows = []
    for d in dialects:
        ranked = (
            sim_matrix[d]
            .drop(d)
            .sort_values(ascending=False)
        )
        for rank, (other, score) in enumerate(ranked.items(), start=1):
            rows.append({'Dialect': d, 'Rank': rank,
                         'Similar To': other, 'Similarity Score': round(score, 4)})
    return pd.DataFrame(rows)

ranking_char     = build_ranking_table(sim_char)
ranking_jaccard  = build_ranking_table(sim_jaccard)
ranking_sentence = build_ranking_table(sim_sentence)
ranking_ensemble = build_ranking_table(sim_ensemble)

# ─────────────────────────────────────────────
# 9.  Print top-5 for each dialect (ensemble)
# ─────────────────────────────────────────────
print("=" * 60)
print("TOP-5 MOST SIMILAR DIALECTS (Ensemble Score)")
print("=" * 60)
for d in dialects:
    top5 = ranking_ensemble[ranking_ensemble['Dialect'] == d].head(5)
    pairs = ", ".join(
        f"{row['Similar To']} ({row['Similarity Score']:.3f})"
        for _, row in top5.iterrows()
    )
    print(f"  {d:<12} → {pairs}")
print()

# ─────────────────────────────────────────────
# 10. Visualisations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(26, 22))
fig.suptitle("Arabic Dialect Similarity Analysis", fontsize=18, fontweight='bold', y=0.98)

heatmap_kwargs = dict(
    annot=True, fmt=".2f", annot_kws={"size": 6.5},
    cmap="YlOrRd", vmin=0, vmax=1,
    linewidths=0.4, linecolor='white'
)

titles = [
    ("Char N-gram TF-IDF Cosine",  sim_char,     axes[0, 0]),
    ("Vocabulary Jaccard",         sim_jaccard,  axes[0, 1]),
    ("Sentence Word-Overlap",      sim_sentence, axes[1, 0]),
    ("Ensemble (Average)",         sim_ensemble, axes[1, 1]),
]

for title, mat, ax in titles:
    sns.heatmap(mat, ax=ax, **heatmap_kwargs)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/user-data/outputs/dialect_heatmaps.png', dpi=150, bbox_inches='tight')
print("Saved: dialect_heatmaps.png")
plt.close()

# ─────────────────────────────────────────────
# 11. Bar chart – top-5 most similar per dialect
#     (ensemble, one subplot per dialect)
# ─────────────────────────────────────────────
n_dialects = len(dialects)
ncols = 4
nrows = -(-n_dialects // ncols)   # ceiling division

fig2, axes2 = plt.subplots(nrows, ncols,
                            figsize=(ncols * 5, nrows * 3.5))
axes2 = axes2.flatten()

palette = sns.color_palette("RdYlGn", 10)

for idx, d in enumerate(dialects):
    ax = axes2[idx]
    top = ranking_ensemble[ranking_ensemble['Dialect'] == d].head(10)
    colors = [palette[i] for i in range(len(top))][::-1]
    bars = ax.barh(top['Similar To'][::-1], top['Similarity Score'][::-1],
                   color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(d, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Similarity', fontsize=8)
    ax.tick_params(labelsize=8)
    for bar, val in zip(bars, top['Similarity Score'][::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=7.5)

# hide unused subplots
for idx in range(n_dialects, len(axes2)):
    axes2[idx].set_visible(False)

fig2.suptitle("Top-10 Most Similar Dialects per Dialect  (Ensemble Score)",
              fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/user-data/outputs/dialect_rankings_bar.png', dpi=150, bbox_inches='tight')
print("Saved: dialect_rankings_bar.png")
plt.close()

# ─────────────────────────────────────────────
# 12. Save rankings to Excel with one sheet each
# ─────────────────────────────────────────────
out_xlsx = '/mnt/user-data/outputs/dialect_similarity_rankings.xlsx'
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    # Full similarity matrices
    sim_ensemble.round(4).to_excel(writer, sheet_name='Ensemble Matrix')
    sim_char.round(4).to_excel(writer,     sheet_name='Char NGram Matrix')
    sim_jaccard.round(4).to_excel(writer,  sheet_name='Jaccard Matrix')
    sim_sentence.round(4).to_excel(writer, sheet_name='Sentence Overlap Matrix')
    # Ranking tables
    ranking_ensemble.to_excel(writer, sheet_name='Rankings (Ensemble)', index=False)
    ranking_char.to_excel(writer,     sheet_name='Rankings (Char NGram)', index=False)
    ranking_jaccard.to_excel(writer,  sheet_name='Rankings (Jaccard)', index=False)
    ranking_sentence.to_excel(writer, sheet_name='Rankings (Sentence)', index=False)

print(f"Saved: dialect_similarity_rankings.xlsx")
print("\nDone ✓")
