"""
Arabic POS Tagging, Analysis & Filler Removal
================================================
Uses a comprehensive rule-based tagger covering:
  PREP  – Prepositions          (من، في، على، مع، إلى …)
  CONJ  – Conjunctions          (و، أو، لكن، ثم …)
  PART  – Particles/Negation    (ما، لا، لم، لن، هل، هو، هي …)
  PRON  – Pronouns              (أنا، أنت، هو، هي، نحن …)
  DET   – Definite article      (ال- prefix)
  VERB  – Verbs                 (pattern + prefix clues)
  NOUN  – Nouns                 (default for unknowns + morphology)
  ADJ   – Adjectives
  ADV   – Adverbs
  NUM   – Numbers
  PUNCT – Punctuation
  INTERJ– Interjections

Filler/function POS removed: PREP, CONJ, PART, PRON, DET
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from collections import Counter, defaultdict
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 1. Rule-Based Arabic POS Tagger
# ══════════════════════════════════════════════════════

# ── Closed-class word lists ──────────────────────────
PREPOSITIONS = {
    'من','في','على','مع','إلى','الى','عن','بين','بعد','قبل','تحت','فوق',
    'خلف','أمام','امام','خلال','حول','ضد','لدى','لدي','عند','عند',
    'داخل','خارج','بجانب','وراء','مقابل','نحو','بدون','دون','رغم',
    'حتى','حتا','مثل','مع','لـ','بـ','كـ','لل','بال','فال','وال',
    'إزاء','ازاء','تجاه','حيال','جهة','ناحية','قدام','قبال','جنب',
    'فوق','تحت','وسط','عبر','ب','ل','ك','لمدة','لحد','لعند','لغاية',
    'لحتى','لين','حتا','كبل','ابيش','بكم','بشقد','شحال','قداش','بقداش',
},
CONJUNCTIONS = {
    'و','أو','او','لكن','بل','ثم','فـ','وال','فال','بينما','حين',
    'عندما','لما','لو','إذا','اذا','إن','ان','أن','لأن','لان','لأنه',
    'لانه','كما','مما','مهما','حيث','حيثما','كيما','كي','لكي',
    'ولكن','وإن','وان','رغم','بدل','عوض','واش','وش','اش',
},
PARTICLES = {
    # negation
    'ما','لا','لم','لن','مش','مو','مو','ماش','ماهو','مهوش','ميش',
    'ماعندي','مافي','ماكو','ماكو','مانه','ماني','مانو','ماش',
    'ماكانش','ماقدرتش','ماسبقليش','ماعمريش',
    # question
    'هل','أ','ما','ماذا','ماشي','ايش','ايه','شو','شنو','واش','آش','اش','وش',
    'كيفاش','كيفش','كيفما','منين','وين','فين','وينك','اين','أين','ليش',
    'ليه','لماذا','لمه','علاش','علاه','علام','علاما','لأيش','شبيه',
    # modal/aspect
    'رح','رح','راح','بدي','بدك','بده','بدها','بدنا','بدكم','بدهم',
    'حيكون','حيروح','لازم','لاازم','لابد','مضطر','ممكن','يمكن',
    'تقدر','يقدر','نقدر','اقدر','بقدر','يستطيع','قادر','يليق',
    # discourse
    'هيا','يلا','يلاه','هيا','اه','آه','ها','هاه','وه','أوه','اوه',
    'اوكي','أوكي','ايوه','ايوا','آيوا','نعم','اي','اه','أه',
    # topic/focus
    'قد','قط','ابد','أبدا','ابدا','دائما','دايما','دايمن','هيك','كذا','كذلك',
    'أيضا','ايضا','كمان','كمن','برضو','برضه','حتى','حتا','اللي','لي',
    'لك','له','لها','لنا','لكم','لهم','إياه','اياه',
    # demonstratives (function-word-like)
    'هذا','هاذا','هاذ','هاد','هادا','هادي','هاد','هذه','هاذه',
    'هذي','هاذي','هذان','هذين','ذلك','ذاك','ذيك','تلك','تيك',
    'هناك','هنيك','هونيك','هناكا','هنا','هون','هوني','هناي','هنايا',
    'ههنا','ثمة','تما','تم','هضا','هادو','هاكا','اهوكا',
},
PRONOUNS = {
    'أنا','انا','أنت','انت','انتي','أنتِ','أنتم','انتم','أنتن','انتن',
    'هو','هي','هم','هن','نحن','احنا','إحنا','حنا','نا',
    'إنت','انتَ','انتِ',
    # object/attached (standalone forms)
    'ياه','ياك','ياها','ياهم','يانا','ياكم',
    'إياي','اياي','إياك','اياك','إياه','اياه','إياها','اياها',
    'إيانا','ايانا','إياكم','اياكم','إياهم','اياهم',
    # dialect forms
    'آنا','أنا','انه','انها','اني','إني','إنه','إنها',
},
NUMBERS = {
    'واحد','اثنين','تلاتة','أربعة','خمسة','ستة','سبعة','ثمانية','تسعة','عشرة',
    'عشرين','ثلاثين','أربعين','خمسين','ستين','سبعين','ثمانين','تسعين','مية','مئة',
    'ألف','مليون','اثنان','ثلاثة','اربعة','خمسه','ستة','سبعة','ثمانيه','تسعه',
    'واحده','تنين','تلتة','اربعه','عشره','خمستاش','عشرتاش','خمستعش',
    'اول','ثاني','ثالث','رابع','خامس','سادس','سابع','ثامن','تاسع','عاشر',
    '0','1','2','3','4','5','6','7','8','9',
    '١','٢','٣','٤','٥','٦','٧','٨','٩','٠',
},
INTERJECTIONS = {
    'يا','ياه','آه','أوف','اوف','عيب','بسم الله','الحمد لله','سبحان الله',
    'إن شاء الله','الله','آمين','يسلمو','شكرا','شكراً','أهلا','مرحبا',
    'أهلا وسهلا','صباح الخير','مساء الخير','يعطيك العافية','يعطيك',
    'الله يعطيك','تفضل','تفضلي','اتفضل','اتفضلي','عفواً','عفوا',
    'لو سمحت','لو سمحتي','من فضلك','بكرا','بكره',
},
ADVERBS = {
    'الآن','الان','هلق','هلأ','هلا','دلوقتي','توا','دابا','دابا','الحين','الحين',
    'الحين','الحلحين','هسع','هسا','هساع','الساعة','هلحين','هلحين',
    'غداً','غدا','بكرا','بكره','بكرة','امبارح','امس','أمس','أمبارح',
    'دائماً','دايما','دايماً','أحياناً','احيانا','نادراً','نادرا',
    'كثيراً','كثيرا','قليلاً','قليلا','شوية','شوي','جداً','جدا',
    'سريعاً','سريعا','ببطء','فوراً','فورا','مباشرةً','مباشرة',
    'دغري','سيدا','نيشان','طوالي','على طول','علطول',
    'أيضاً','أيضا','كمان','برضو','برضه','كذلك','أيضاً',
    'فقط','بس','رقط','غير','إلا','الا','حتى',
    'هنا','هون','هوني','هناك','هنيك','هونيك','تما',
    'اليوم','النهارده','النهار','اليوم','هاليوم',
},

# flatten tuples → sets (they were grouped with trailing commas)
PREPOSITIONS  = PREPOSITIONS[0]
CONJUNCTIONS  = CONJUNCTIONS[0]
PARTICLES     = PARTICLES[0]
PRONOUNS      = PRONOUNS[0]
NUMBERS       = NUMBERS[0]
INTERJECTIONS = INTERJECTIONS[0]
ADVERBS       = ADVERBS[0]

# ── Verb morphology helpers ──────────────────────────
# Present-tense prefixes
VERB_PREF = re.compile(r'^[بتينأي][ـ]?[^\s]{2,}')
# Past-tense verb endings
VERB_SUFF_PAST = re.compile(r'.*[تنوا]$')
# Common verb templates (root patterns mapped onto ف ع ل slots)
VERB_PATTERNS = re.compile(
    r'^(يقدر|تقدر|يقدرو|تقدرو|نقدر|اقدر|بقدر|قادر|'
    r'يروح|تروح|روح|راح|يجي|تجي|جي|جيت|'
    r'يشوف|تشوف|شاف|شفت|شفت|'
    r'يعطي|تعطي|اعطي|اعطاه|'
    r'يسمع|تسمع|سمع|سمعت|'
    r'يعمل|تعمل|عمل|عملت|'
    r'يكون|تكون|كان|كانت|كانو|'
    r'ياخذ|تاخذ|أخذ|اخذ|خذ|'
    r'يعرف|تعرف|عرف|عرفت|'
    r'يقول|تقول|قال|قلت|'
    r'يبغى|بغيت|بغى|'
    r'يشتري|اشترى|اشتريت|'
    r'ينام|نام|نمت|'
    r'يمشي|مشى|مشيت|امشي|'
    r'يطلع|طلع|طلعت|اطلع|'
    r'يدخل|دخل|دخلت|'
    r'يخرج|خرج|خرجت|'
    r'يكتب|كتب|كتبت|'
    r'يقرأ|قرأ|قرأت|'
    r'يأكل|أكل|اكل|اكلت|'
    r'يشرب|شرب|شربت|'
    r'ينزل|نزل|نزلت|'
    r'يصل|وصل|وصلت|'
    r'يبدأ|بدأ|بدأت|'
    r'ينتظر|انتظر|انتظرت|'
    r'يحتاج|احتاج|احتجت|'
    r'يحب|أحب|احب|حبيت|'
    r'يساعد|ساعد|ساعدت|'
    r'يبدل|بدل|بدلت|'
    r'يوجد|وجد|وجدت|موجود|'
    r'يعيش|عاش|عشت|'
    r'يسكن|سكن|سكنت|'
    r'يستطيع|استطاع|استطعت|'
    r'يمكن|امكن|'
    r'يسأل|سأل|سألت|'
    r'يجلس|جلس|جلست|'
    r'يفضل|فضل|فضلت|'
    r'يستأجر|استأجر|استأجرت|'
    r'يحجز|حجز|حجزت|'
    r'يدفع|دفع|دفعت|'
    r'يبيع|باع|بعت|'
    r'يطلب|طلب|طلبت|'
    r'يفتح|فتح|فتحت|'
    r'يغلق|اغلق|أغلق|غلقت|'
    r'يوصل|وصّل|وصلت|'
    r'يرجع|رجع|رجعت|'
    r'يجيب|جاب|جبت)$'
)

# ── Core tagging function ────────────────────────────
def tag_word(word):
    """Return a POS tag for a single Arabic word token."""
    w = word.strip()

    # Punctuation
    if re.fullmatch(r'[.,،؟?!؛;:\-–—()«»"\'…]+', w):
        return 'PUNCT'

    # Digits / numbers
    if re.fullmatch(r'[\d٠-٩]+', w):
        return 'NUM'
    if w in NUMBERS:
        return 'NUM'

    # Closed-class lookups (most specific first)
    if w in PRONOUNS:   return 'PRON'
    if w in PARTICLES:  return 'PART'
    if w in CONJUNCTIONS: return 'CONJ'
    if w in PREPOSITIONS: return 'PREP'
    if w in ADVERBS:    return 'ADV'
    if w in INTERJECTIONS: return 'INTERJ'

    # Definite article attached to word
    if w.startswith('ال') and len(w) > 3:
        return 'NOUN'   # ال + noun is a definite noun

    # Known verb patterns
    if VERB_PATTERNS.match(w):
        return 'VERB'

    # Present-tense prefix clues (ي/ت/ن/ا for 3rd/2nd/1st person)
    if re.match(r'^[يتنا].{3,}$', w) and not w.startswith('ال'):
        if not w.startswith('ان') and not w.startswith('است'):
            return 'VERB'

    # Imperative verb (2-char root + imperative structure)
    if re.match(r'^[اإ].{2,4}$', w):
        return 'VERB'

    # Taa marbuta ending → strong noun/adjective signal
    if w.endswith('ة') or w.endswith('ه') and len(w) > 3:
        return 'NOUN'

    # Broken plural / adjective patterns (difficult; default NOUN)
    # Past-tense endings (-ت, -نا, -تم, -وا)
    if re.match(r'.{3,}[تن]$', w) or w.endswith('وا') or w.endswith('ون'):
        return 'VERB'

    # Masdar / verbal noun patterns
    if re.match(r'^(است|انت|مت|مست).{3,}$', w):
        return 'NOUN'

    # Default: NOUN (catches content words, proper nouns, unknowns)
    return 'NOUN'


def tag_sentence(sentence):
    """Return list of (word, POS) tuples for a sentence."""
    if pd.isna(sentence):
        return []
    tokens = str(sentence).split()
    return [(tok, tag_word(tok)) for tok in tokens]


def pos_counts_for_sentence(sentence):
    """Return Counter of POS tags in a sentence."""
    return Counter(tag for _, tag in tag_sentence(sentence))


# ══════════════════════════════════════════════════════
# 2. Load data & run POS tagging
# ══════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv('/mnt/user-data/uploads/MADAR_combined_Arabic_2_.csv')
dialects = [c for c in df.columns if c != 'ID']

print("Tagging all sentences (this takes a moment) …")

# Global POS counter and per-dialect counter
global_counter = Counter()
dialect_counters = {d: Counter() for d in dialects}
total_tokens = 0

for d in dialects:
    for sent in df[d].dropna():
        tags = tag_sentence(sent)
        for _, pos in tags:
            global_counter[pos] += 1
            dialect_counters[d][pos] += 1
            total_tokens += 1

print(f"Tagged {total_tokens:,} tokens across {len(dialects)} dialects.\n")

# ── Global POS distribution ──────────────────────────
print("═" * 50)
print("GLOBAL POS DISTRIBUTION")
print("═" * 50)
for pos, cnt in global_counter.most_common():
    pct = 100 * cnt / total_tokens
    print(f"  {pos:<8} {cnt:>8,}  ({pct:5.1f}%)")
print()

# ── Per-dialect POS distribution ────────────────────
print("═" * 50)
print("POS DISTRIBUTION PER DIALECT (top 3 tags)")
print("═" * 50)
for d in dialects:
    c = dialect_counters[d]
    total = sum(c.values())
    top3 = ", ".join(f"{p}:{100*n/total:.0f}%" for p,n in c.most_common(3))
    print(f"  {d:<14}  {top3}")
print()

# ══════════════════════════════════════════════════════
# 3. Identify & remove filler POS
# ══════════════════════════════════════════════════════
# Filler = function words with low semantic content:
# PREP, CONJ, PART, PRON, PUNCT
FILLER_POS = {'PREP', 'CONJ', 'PART', 'PRON', 'PUNCT'}
print(f"Filler POS to remove: {FILLER_POS}")
filler_total = sum(global_counter[p] for p in FILLER_POS)
print(f"  → {filler_total:,} tokens removed ({100*filler_total/total_tokens:.1f}% of all tokens)\n")

def remove_fillers(sentence):
    """Remove filler-tagged tokens from a sentence."""
    if pd.isna(sentence):
        return np.nan
    tokens = str(sentence).split()
    kept = [tok for tok in tokens if tag_word(tok) not in FILLER_POS]
    return " ".join(kept) if kept else np.nan

print("Cleaning dataset …")
df_clean = df.copy()
for d in dialects:
    df_clean[d] = df[d].apply(remove_fillers)

# ── Stats on cleaning ────────────────────────────────
orig_words  = sum(len(str(s).split()) for d in dialects for s in df[d].dropna())
clean_words = sum(len(str(s).split()) for d in dialects for s in df_clean[d].dropna())
print(f"Original total words : {orig_words:,}")
print(f"After removal        : {clean_words:,}")
print(f"Removed              : {orig_words - clean_words:,} ({100*(orig_words-clean_words)/orig_words:.1f}%)\n")

# ══════════════════════════════════════════════════════
# 4. Visualisations
# ══════════════════════════════════════════════════════
POS_COLORS = {
    'NOUN':'#4C72B0','VERB':'#DD8452','ADJ':'#55A868',
    'PREP':'#C44E52','CONJ':'#8172B2','PART':'#937860',
    'PRON':'#DA8BC3','ADV':'#8C8C8C','NUM':'#CCB974',
    'PUNCT':'#64B5CD','INTERJ':'#E377C2','DET':'#17BECF',
}

# ── Fig 1: Global POS pie + bar ──────────────────────
fig1, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(16, 7))
fig1.suptitle("Arabic Dialect Corpus – POS Distribution", fontsize=16, fontweight='bold')

labels = [p for p, _ in global_counter.most_common()]
sizes  = [global_counter[p] for p in labels]
colors = [POS_COLORS.get(p, '#aaaaaa') for p in labels]

ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=140, pctdistance=0.82,
           textprops={'fontsize': 10})
ax_pie.set_title("Global Token Share", fontsize=12, fontweight='bold')

# bar
pcts = [100 * s / total_tokens for s in sizes]
bars = ax_bar.barh(labels[::-1], pcts[::-1], color=colors[::-1], edgecolor='white')
ax_bar.set_xlabel("% of Total Tokens", fontsize=11)
ax_bar.set_title("Token Frequency by POS", fontsize=12, fontweight='bold')
for bar, pct in zip(bars, pcts[::-1]):
    ax_bar.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va='center', fontsize=9)
ax_bar.spines[['top','right']].set_visible(False)
ax_bar.set_xlim(0, max(pcts) * 1.18)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pos_global.png', dpi=150, bbox_inches='tight')
print("Saved: pos_global.png")
plt.close()

# ── Fig 2: Stacked bar per dialect ──────────────────
all_pos = [p for p, _ in global_counter.most_common()]
dial_pct = {}
for d in dialects:
    total = sum(dialect_counters[d].values())
    dial_pct[d] = {p: 100 * dialect_counters[d].get(p, 0) / total for p in all_pos}

df_pos = pd.DataFrame(dial_pct, index=all_pos).T   # dialects × POS

fig2, ax = plt.subplots(figsize=(20, 8))
bottom = np.zeros(len(dialects))
x = np.arange(len(dialects))

for pos in all_pos:
    vals = df_pos[pos].values
    ax.bar(x, vals, bottom=bottom, label=pos,
           color=POS_COLORS.get(pos, '#aaaaaa'), edgecolor='white', linewidth=0.3)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(dialects, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("% of Tokens", fontsize=11)
ax.set_title("POS Composition per Dialect", fontsize=14, fontweight='bold')
ax.legend(title='POS', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, 105)

# shade filler POS in legend
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pos_per_dialect.png', dpi=150, bbox_inches='tight')
print("Saved: pos_per_dialect.png")
plt.close()

# ── Fig 3: Filler vs Content split per dialect ──────
fig3, ax3 = plt.subplots(figsize=(16, 6))
filler_pcts, content_pcts = [], []
for d in dialects:
    total = sum(dialect_counters[d].values())
    filler = sum(dialect_counters[d].get(p, 0) for p in FILLER_POS)
    filler_pcts.append(100 * filler / total)
    content_pcts.append(100 * (total - filler) / total)

x = np.arange(len(dialects))
ax3.bar(x, content_pcts, label='Content words (kept)', color='#4C72B0', edgecolor='white')
ax3.bar(x, filler_pcts, bottom=content_pcts, label='Filler words (removed)', color='#C44E52',
        edgecolor='white', alpha=0.85)

ax3.set_xticks(x)
ax3.set_xticklabels(dialects, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel("% of Tokens", fontsize=11)
ax3.set_title("Content vs Filler Token Split per Dialect", fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.spines[['top','right']].set_visible(False)
for i, (fp, cp) in enumerate(zip(filler_pcts, content_pcts)):
    ax3.text(i, 101, f"{fp:.0f}%", ha='center', fontsize=7.5, color='#C44E52', fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pos_filler_split.png', dpi=150, bbox_inches='tight')
print("Saved: pos_filler_split.png")
plt.close()

# ── Fig 4: Heatmap of POS% across dialects ──────────
fig4, ax4 = plt.subplots(figsize=(18, 7))
heat_data = df_pos[all_pos].T    # POS × dialects
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap='YlOrRd',
            linewidths=0.4, linecolor='white',
            annot_kws={"size": 7.5}, ax=ax4)
ax4.set_title("POS Tag % Heatmap (rows=POS, cols=Dialect)", fontsize=13, fontweight='bold')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pos_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: pos_heatmap.png")
plt.close()

# ══════════════════════════════════════════════════════
# 5. Save outputs
# ══════════════════════════════════════════════════════
# Cleaned CSV
df_clean.to_csv('/mnt/user-data/outputs/MADAR_cleaned_no_fillers.csv', index=False)
print("Saved: MADAR_cleaned_no_fillers.csv")

# Excel with POS stats + cleaned data
out_xlsx = '/mnt/user-data/outputs/pos_analysis.xlsx'
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    # Global POS table
    global_df = pd.DataFrame([
        {'POS': p, 'Count': cnt, 'Percentage': f"{100*cnt/total_tokens:.2f}%",
         'Is Filler': 'YES' if p in FILLER_POS else 'no'}
        for p, cnt in global_counter.most_common()
    ])
    global_df.to_excel(writer, sheet_name='Global POS Stats', index=False)

    # Per-dialect POS counts
    dial_counts_df = pd.DataFrame({
        d: {p: dialect_counters[d].get(p, 0) for p in all_pos}
        for d in dialects
    }).T
    dial_counts_df.to_excel(writer, sheet_name='POS Counts per Dialect')

    # Per-dialect POS percentages
    df_pos.round(2).to_excel(writer, sheet_name='POS Pct per Dialect')

    # Removal summary
    summary = pd.DataFrame([
        {'Dialect': d,
         'Original Tokens': sum(dialect_counters[d].values()),
         'Filler Tokens': sum(dialect_counters[d].get(p, 0) for p in FILLER_POS),
         'Filler %': f"{100*sum(dialect_counters[d].get(p,0) for p in FILLER_POS)/max(sum(dialect_counters[d].values()),1):.1f}%"}
        for d in dialects
    ])
    summary.to_excel(writer, sheet_name='Filler Removal Summary', index=False)

print(f"Saved: pos_analysis.xlsx")
print("\n✓ All done.")
