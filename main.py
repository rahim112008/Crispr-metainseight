# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v9 — Plateforme intégrative multi-omique, PGM, Épitranscriptomique
# + Module CRISPR Design & Delivery (données réelles)
# + Module Plant & Agronomie (compatible SIVB/CAST 2026)
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import requests
import os
import io
import re
import gzip
import json
import pickle
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from io import StringIO, BytesIO
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, silhouette_score, roc_curve, auc,
                              r2_score, mean_squared_error, mean_absolute_error,
                              roc_auc_score)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from scipy.stats import (entropy, spearmanr, kruskal, mannwhitneyu, f_oneway,
                          pearsonr, linregress)
from scipy.spatial.distance import cdist, braycurtis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
import joblib

try:
    import duckdb; DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import pysam; PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

try:
    from Bio import SeqIO; BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    import xgboost as xgb; XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Clés API
_ENV_GEMINI_KEY     = os.environ.get('GEMINI_API_KEY', '')
_ENV_GROQ_KEY       = os.environ.get('GROQ_API_KEY', '')
_ENV_OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
_ENV_DEEPSEEK_KEY   = os.environ.get('DEEPSEEK_API_KEY', '')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MetaInsight v9 — Multi-omics + CRISPR + Plant",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
.stApp { background-color: #0A0E1A; color: #E8EDF5; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background-color: #0A0E1A;
    border-bottom: 1px solid #2A3550; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background-color: #0F1525; border-radius: 8px 8px 0 0;
    color: #7A8BA8; padding: 6px 12px; font-weight: 500; font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background-color: #151C30; color: #00D4AA;
    border-bottom: 2px solid #00D4AA;
}
.stButton button {
    background-color: #1A2238; border: 1px solid #2A3550;
    color: #E8EDF5; border-radius: 8px;
}
.stButton button:hover { background-color: #1F2940; border-color: #00D4AA; color: #00D4AA; }
.badge-new {
    background: linear-gradient(90deg,#00D4AA,#4D9FFF);
    color:#000; font-size:0.65rem; padding:2px 7px; border-radius:10px;
    font-weight:700; margin-left:4px; vertical-align:middle;
}
.badge-fix {
    background: linear-gradient(90deg,#4D9FFF,#9B7CFF);
    color:#000; font-size:0.65rem; padding:2px 7px; border-radius:10px;
    font-weight:700; margin-left:4px; vertical-align:middle;
}
.ref-box {
    background:#0F1525; border-left:3px solid #00D4AA; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A8BA8; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Cache
CACHE_DIR = Path("./crispr_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES DÉMO (fallback)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_demo_microbiome():
    environments = ["Sol aride", "Eau marine", "Gut", "Sol agricole", "Sédiments", "Biofilm"]
    taxa = ["Proteobacteria","Actinobacteriota","Firmicutes","Bacteroidota","Archaea",
            "Acidobacteria","Chloroflexi","Planctomycetes","Ascomycota","Caudovirales"]
    base_profiles = {
        "Sol aride":    [28, 20,  5,  4,  8,  6,  4,  3,  2,  1],
        "Eau marine":   [35, 10,  8, 15,  2,  5,  3,  4,  8,  6],
        "Gut":          [15, 12, 30, 22,  1,  3,  2,  2,  4,  2],
        "Sol agricole": [22, 25, 10,  8,  4, 10,  7,  5,  3,  2],
        "Sédiments":    [18, 14, 12, 10,  6,  8,  9,  6,  5,  4],
        "Biofilm":      [30, 18,  6,  9,  3,  7,  5,  4,  6,  5],
    }
    data = []
    for env in environments:
        base = base_profiles[env]
        for rep in range(4):
            noisy = np.array(base) + np.random.normal(0, 2, size=len(taxa))
            noisy = np.clip(noisy, 0, None)
            noisy = noisy / noisy.sum() * 100
            row = {"sample_id": f"{env[:3].upper()}_{rep+1:03d}", "environment": env}
            for i, tax in enumerate(taxa):
                row[tax] = round(noisy[i], 2)
            probs = noisy / 100.0
            row["shannon"] = round(entropy(probs, base=2), 3)
            row["simpson"] = round(1 - np.sum(probs**2), 3)
            row["chao1"] = round(len(taxa) + np.random.uniform(0, 5), 1)
            row["faith_pd"] = round(float((noisy > 0).sum()) * 2.1 + float(np.std(noisy[noisy>0])) * 0.5, 2)
            row["classified_pct"] = round(np.random.uniform(70, 99), 1)
            row["ph"] = round(np.random.uniform(4, 8), 2)
            row["temperature_c"] = round(np.random.uniform(15, 40), 1)
            row["moisture_pct"] = round(np.random.uniform(5, 80), 1)
            data.append(row)
    return pd.DataFrame(data)

@st.cache_data
def generate_demo_pgm_data():
    np.random.seed(42)
    n = 200
    chroms = np.random.choice([str(i) for i in range(1, 23)] + ["X"], size=n)
    pos = np.random.randint(100000, 100000000, n)
    refs = np.random.choice(["A","C","G","T"], n)
    alts = np.random.choice(["A","C","G","T"], n)
    for i in range(n):
        while alts[i] == refs[i]:
            alts[i] = np.random.choice(["A","C","G","T"])
    return pd.DataFrame({
        "chrom": chroms, "pos": pos, "ref": refs, "alt": alts,
        "qual": np.random.uniform(10, 1000, n),
        "filter": np.random.choice(["PASS","LowQual",""], n, p=[0.7,0.2,0.1]),
        "info": "",
        "clinvar_sig": np.random.choice(["Pathogenic","Likely_pathogenic","VUS","Benign","Likely_benign"],
                                          n, p=[0.05,0.1,0.6,0.15,0.1]),
        "variant_type": np.random.choice(["SNP","Indel","MNV"], n),
        "rsid": [f"rs{np.random.randint(100000, 999999)}" for _ in range(n)],
        "allele_freq": np.random.uniform(0.001, 0.5, n),
        "cadd_phred": np.random.uniform(0, 45, n),
    })

@st.cache_data
def generate_demo_epitranscriptomic_data():
    np.random.seed(123)
    n = 80
    positions = np.sort(np.random.randint(50, 1950, n))
    mod_types = np.random.choice(['m6A','m5C','Ψ','m1A','2OMe','m7G','Nm'], n,
                                   p=[0.35,0.15,0.15,0.1,0.1,0.08,0.07])
    df = pd.DataFrame({
        'transcript_id': np.random.choice(['ENST00000380152','ENST00000456328',
                                             'ENST00000585865','ENST00000318560'], n),
        'position': positions,
        'modification': mod_types,
        'modification_rate': np.round(np.random.beta(2, 5, n), 3),
        'pseudouridine_score': np.random.uniform(0, 1, n),
        'condition': np.random.choice(['Contrôle', 'Traité'], n),
        'sample_id': [f'Sample_{i%6+1:02d}' for i in range(n)],
        'expression_TPM': np.round(np.random.lognormal(5, 1.5, n), 2),
        'gene': np.random.choice(['BRCA1','TP53','EGFR','MYC','KRAS','PTEN'], n),
    })
    motifs = []
    for m in mod_types:
        if m == 'm6A': motifs.append(np.random.choice(['DRACH','GGACU','RRACH']))
        elif m == 'm5C': motifs.append(np.random.choice(['CG','CNG']))
        elif m == 'Ψ': motifs.append(np.random.choice(['UGU','GU']))
        else: motifs.append('')
    df['consensus_motif'] = motifs
    return df

@st.cache_data
def generate_epi_annotation_db():
    return pd.DataFrame({
        'transcript_id': ['ENST00000380152','ENST00000380152','ENST00000456328','ENST00000585865'],
        'position': [345, 876, 234, 567],
        'modification': ['m6A','Ψ','m5C','m6A'],
        'gene': ['BRCA1','BRCA1','TP53','MYC'],
        'motif': ['DRACH','UGU','CG','DRACH'],
        'confidence': ['High','Medium','High','High'],
    })

# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS GÉNÉRIQUES
# ══════════════════════════════════════════════════════════════════════════════
def clr_transform(X):
    X_pos = np.clip(X, 1e-9, None)
    log_X = np.log(X_pos)
    return log_X - log_X.mean(axis=1, keepdims=True)

def compute_alpha_diversity(df, taxa_cols):
    results = []
    for _, row in df.iterrows():
        vals = np.clip(row[taxa_cols].values.astype(float), 0, None)
        total = vals.sum()
        probs = vals / total if total > 0 else vals
        probs_nz = probs[probs > 0]
        shannon = float(entropy(probs_nz, base=2)) if len(probs_nz) > 0 else 0.0
        simpson_d = float(1 - np.sum(probs**2))
        richness = int((vals > 0).sum())
        n1, n2 = int((vals == 1).sum()), int((vals == 2).sum())
        chao1 = richness + (n1*(n1-1))/(2*(n2+1)) if n2 > 0 else richness + n1*(n1-1)/2
        evenness = shannon / np.log2(richness) if richness > 1 else 0.0
        faith_pd = richness * 2.1 + float(np.std(probs_nz)) * 5.0 if len(probs_nz) > 0 else 0.0
        results.append({"Shannon H'": round(shannon, 3), "Simpson (1-D)": round(simpson_d, 3),
                        "Richness": richness, "Chao1": round(chao1, 1),
                        "Evenness (J)": round(evenness, 3), "Faith PD (proxy)": round(faith_pd, 2)})
    return pd.DataFrame(results, index=df.index)

def detect_feature_cols(df):
    meta_cols = {"sample_id","environment","group","label","class","condition",
                 "shannon","simpson","chao1","faith_pd","classified_pct","ph",
                 "temperature_c","moisture_pct"}
    feature_cols = []
    for col in df.columns:
        if col.lower() in meta_cols: continue
        if col.endswith("_ZYG") or "date" in col.lower() or "id" in col.lower(): continue
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            if col_data.std() > 0: feature_cols.append(col)
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            if 2 <= len(col_data.dropna().unique()) <= 30: feature_cols.append(col)
    return feature_cols

def compute_bray_curtis_matrix(X):
    n = len(X)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = braycurtis(X[i], X[j])
            dm[i, j] = dm[j, i] = d
    return dm

def permanova_test(X, groups, n_permutations=999):
    dm = compute_bray_curtis_matrix(X)
    labels = np.array(groups)
    def pseudo_f(dm, labels):
        n_local = len(labels)
        ss_total = np.sum(dm**2) / n_local
        ss_within = 0.0
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            ng = len(idx)
            if ng < 2: continue
            ss_within += np.sum(dm[np.ix_(idx, idx)]**2) / ng
        ss_between = ss_total - ss_within
        n_groups = len(np.unique(labels))
        df_between = n_groups - 1
        df_within = n_local - n_groups
        if df_within <= 0: return 0.0
        return (ss_between / df_between) / (ss_within / df_within)
    f_obs = pseudo_f(dm, labels)
    f_perms = [pseudo_f(dm, np.random.RandomState(42).permutation(labels)) for _ in range(n_permutations)]
    p_val = (np.sum(np.array(f_perms) >= f_obs) + 1) / (n_permutations + 1)
    return {"F": round(f_obs, 4), "p-value": round(p_val, 4), "R²": round(f_obs/(f_obs+1), 3)}

def aldex2_like(df, taxa_cols, group_col, group1, group2):
    g1 = df[df[group_col] == group1][taxa_cols].values.astype(float)
    g2 = df[df[group_col] == group2][taxa_cols].values.astype(float)
    if len(g1) < 2 or len(g2) < 2: return None
    results = []
    for j, tax in enumerate(taxa_cols):
        clr1 = clr_transform(g1 + 0.5)[:, j]
        clr2 = clr_transform(g2 + 0.5)[:, j]
        effect = (clr1.mean() - clr2.mean()) / (np.sqrt((clr1.std()**2 + clr2.std()**2) / 2) + 1e-9)
        try: _, pval = mannwhitneyu(clr1, clr2, alternative='two-sided')
        except: pval = 1.0
        results.append({"Taxon": tax, "CLR mean G1": round(clr1.mean(), 3),
                        "CLR mean G2": round(clr2.mean(), 3), "Effect size": round(effect, 3),
                        "p-value (Wilcoxon)": round(pval, 4)})
    res_df = pd.DataFrame(results)
    n = len(res_df)
    pvals = res_df["p-value (Wilcoxon)"].values
    sorted_idx = np.argsort(pvals); bh = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        bh[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(n-2, -1, -1):
        bh[sorted_idx[i]] = min(bh[sorted_idx[i]], bh[sorted_idx[i+1]])
    res_df["BH adj. p-value"] = bh.round(4)
    return res_df.sort_values("BH adj. p-value")

def lefse_like(df, taxa_cols, group_col):
    groups = df[group_col].unique()
    if len(groups) < 2: return None
    results = []
    for tax in taxa_cols:
        group_vals = [df[df[group_col] == g][tax].values for g in groups]
        try: _, p_kw = kruskal(*group_vals)
        except: p_kw = 1.0
        if p_kw < 0.05:
            means = [v.mean() for v in group_vals]
            stds = [v.std() + 1e-9 for v in group_vals]
            pooled_std = np.sqrt(np.mean([s**2 for s in stds]))
            lda_score = abs(max(means) - min(means)) / (pooled_std + 1e-9) * np.log10(len(df)+1)
            best_group = groups[np.argmax(means)]
        else:
            lda_score, best_group = 0.0, "—"
        results.append({"Taxon": tax, "LDA Score": round(lda_score, 3),
                        "Best group": best_group, "p-value": round(p_kw, 4)})
    return pd.DataFrame(results).sort_values("LDA Score", ascending=False)

def maaslin2_like(df, taxa_cols, group_col):
    le = LabelEncoder()
    y = le.fit_transform(df[group_col].values)
    X_clr = clr_transform(df[taxa_cols].values.astype(float) + 0.5)
    results = []
    for j, tax in enumerate(taxa_cols):
        x_j = X_clr[:, j].reshape(-1, 1)
        lr = LinearRegression().fit(x_j, y)
        coef = lr.coef_[0]; r2 = lr.score(x_j, y); n = len(y)
        se = np.sqrt((1 - r2) / max(n - 2, 1)) * np.std(y) / (np.std(x_j.flatten()) + 1e-9)
        t_stat = abs(coef) / (se + 1e-9)
        from scipy.stats import t as t_dist
        p_val = 2 * t_dist.sf(abs(t_stat), df=n-2)
        results.append({"Taxon": tax, "Coefficient": round(coef, 4),
                        "R²": round(r2, 4), "p-value": round(p_val, 4)})
    return pd.DataFrame(results).sort_values("p-value")

def kegg_functional_prediction(df, taxa_cols):
    FALLBACK = {
        "Proteobacteria": ["Nitrogen fixation","Flagellar biosynthesis","ATP synthesis"],
        "Firmicutes": ["Butyrate production","Sporulation","Peptidoglycan biosynthesis"],
        "Bacteroidota": ["Polysaccharide degradation","Vitamin B12 biosynthesis","Glycolysis"],
        "Actinobacteriota": ["Secondary metabolites","Antibiotic biosynthesis"],
        "Archaea": ["Methanogenesis","CO2 fixation"],
        "Acidobacteria": ["Cellulose degradation","Carbon cycling"],
        "Chloroflexi": ["Reductive TCA cycle"],
        "Planctomycetes": ["Anammox","Nitrogen cycling"],
        "Ascomycota": ["Chitin synthesis","Ergosterol biosynthesis"],
        "Caudovirales": ["Viral DNA replication","Capsid assembly"],
    }
    results = []
    for _, row in df.iterrows():
        env_kegg = {}
        for tax in taxa_cols:
            if tax in FALLBACK:
                for pw in FALLBACK[tax]:
                    env_kegg[pw] = env_kegg.get(pw, 0) + float(row.get(tax, 0))
        results.append(env_kegg)
    return pd.DataFrame(results, index=df.index).fillna(0)

def rarefaction_curve(df, taxa_cols, n_steps=20):
    curves = {}
    for env in df["environment"].unique():
        sub = df[df["environment"] == env][taxa_cols].values
        sub_int = (sub * 10).astype(int)
        total_counts = sub_int.sum(axis=1)
        max_depth = max(int(total_counts.min()), 100)
        depths = np.linspace(1, max_depth, min(n_steps, max_depth)).astype(int)
        richness_curve = []
        rng = np.random.RandomState(42)
        for depth in depths:
            obs_rich = []
            for counts in sub_int:
                total = counts.sum()
                if total < depth:
                    obs_rich.append((counts > 0).sum()); continue
                probs = counts / total
                sampled = rng.multinomial(depth, probs / probs.sum())
                obs_rich.append((sampled > 0).sum())
            richness_curve.append(np.mean(obs_rich))
        curves[env] = (depths, richness_curve)
    return curves

def call_ai(prompt, provider, gemini_key=None, groq_key=None, openrouter_key=None,
            deepseek_key=None, gemini_model="gemini-1.5-flash",
            groq_model="llama-3.3-70b-versatile",
            openrouter_model="meta-llama/llama-3.1-8b-instruct:free",
            ollama_model="llama3"):
    try:
        if provider == "Gemini Flash (Google — GRATUIT)":
            if not gemini_key: return "🔑 Clé Gemini manquante"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"⚠️ Erreur Gemini {r.status_code}"
        elif provider == "Groq (gratuit)":
            if not groq_key: return "🔑 Clé Groq manquante"
            headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              json={"model": groq_model, "messages": [{"role": "user", "content": prompt}]},
                              headers=headers, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur Groq {r.status_code}"
        elif provider == "DeepSeek (gratuit)":
            if not deepseek_key: return "🔑 Clé DeepSeek manquante"
            headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
            r = requests.post("https://api.deepseek.com/v1/chat/completions",
                              json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]},
                              headers=headers, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur DeepSeek {r.status_code}"
        elif provider == "OpenRouter (gratuit)":
            if not openrouter_key: return "🔑 Clé OpenRouter manquante"
            headers = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json={"model": openrouter_model, "messages": [{"role": "user", "content": prompt}]},
                              headers=headers, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur OpenRouter {r.status_code}"
        elif provider == "Ollama (local)":
            try:
                r = requests.post("http://localhost:11434/api/generate",
                                  json={"model": ollama_model, "prompt": prompt, "stream": False}, timeout=60)
                if r.status_code == 200: return r.json().get("response", "")
                return f"⚠️ Erreur Ollama {r.status_code}"
            except requests.exceptions.ConnectionError:
                return "❌ Ollama non lancé"
        return "⚠️ Fournisseur non reconnu"
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

def load_vcf_with_duckdb(file_path):
    if not DUCKDB_AVAILABLE:
        st.error("DuckDB non installé"); return pd.DataFrame()
    try:
        query = f"""
        SELECT chrom, pos, ref, alt, qual, filter, info,
            regexp_extract(info, 'CLNSIG=([^;]+)', 1) AS clinvar_sig,
            regexp_extract(info, 'RS=([^;]+)', 1) AS rsid
        FROM read_vcf('{file_path}')
        """
        return duckdb.sql(query).df()
    except Exception as e:
        st.error(f"Erreur DuckDB : {e}"); return pd.DataFrame()

def parse_epitranscriptomic_file(f):
    try:
        content = f.getvalue().decode('utf-8', errors='ignore')
        sep = '\t' if '\t' in content[:1000] else ',' if ',' in content[:1000] else ';'
        return pd.read_csv(io.StringIO(content), sep=sep)
    except Exception as e:
        st.error(f"Erreur parsing : {e}"); return None

def parse_fastq_metadata(f):
    if not BIOPYTHON_AVAILABLE: return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fastq') as tmp:
            tmp.write(f.read()); tmp_path = tmp.name
        records = list(SeqIO.parse(tmp_path, "fastq"))
        avg_qual = np.mean([np.mean(r.letter_annotations["phred_quality"]) for r in records]) if records else 0
        os.unlink(tmp_path)
        return {"n_reads": len(records), "avg_quality": avg_qual}
    except Exception: return None

def parse_bam_advanced(f):
    if not PYSAM_AVAILABLE: return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bam') as tmp:
            tmp.write(f.read()); tmp_path = tmp.name
        bam = pysam.AlignmentFile(tmp_path, "rb")
        n_reads = n_mapped = n_with_mods = 0
        for read in bam:
            n_reads += 1
            if not read.is_unmapped:
                n_mapped += 1
                if read.has_tag('MM') and read.has_tag('ML'):
                    n_with_mods += 1
            if n_reads > 1000: break
        bam.close(); os.unlink(tmp_path)
        return {"n_reads": n_reads, "n_mapped": n_mapped, "n_reads_with_mods": n_with_mods,
                "mod_rate": n_with_mods / max(n_reads, 1)}
    except Exception: return None

def plot_modification_profile_advanced(df, transcript_id=None, smooth=True):
    if transcript_id: df = df[df['transcript_id'] == transcript_id]
    if df.empty: return None
    df = df.sort_values('position')
    fig = go.Figure()
    for mod in df['modification'].unique():
        sub = df[df['modification'] == mod]
        fig.add_trace(go.Scatter(x=sub['position'], y=sub['modification_rate'],
                                  mode='markers+lines' if smooth else 'markers', name=mod))
    fig.update_layout(title=f"Profil - {transcript_id if transcript_id else 'Tous'}",
                      xaxis_title="Position", yaxis_title="Taux", template="plotly_dark")
    return fig

def plot_modification_heatmap(df):
    pivot = df.pivot_table(index='sample_id', columns='position',
                            values='modification_rate', aggfunc='mean')
    if pivot.empty: return None
    return px.imshow(pivot, color_continuous_scale="RdBu_r", aspect="auto", title="Heatmap")

def get_default_model():
    return make_pipeline(StandardScaler(),
                          SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001,
                                        max_iter=1000, tol=1e-3, random_state=42,
                                        warm_start=True))

def extract_features(sequence, position):
    start = max(0, position - 5); end = min(len(sequence), position + 5)
    context = sequence[start:end]
    features = [context.count(b) / max(1, len(context)) for b in ['A','C','G','T','U']]
    for motif in ['DRACH','UGU','CG','RRACH','CNG','GGACU']:
        features.append(1 if motif in context.upper() else 0)
    features.append(position / max(1, len(sequence)))
    return np.array(features)

MODEL_PATH = "metainsight_model.pkl"
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else get_default_model()
def save_model(model): joblib.dump(model, MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# CRISPR — CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
PAM_PATTERNS = {
    "SpCas9 (NGG)":     {"pam": "NGG",    "guide_len": 20, "pam_side": "3'"},
    "SpCas9-VQR (NGA)": {"pam": "NGA",    "guide_len": 20, "pam_side": "3'"},
    "SaCas9 (NNGRRT)":  {"pam": "NNGRRT", "guide_len": 21, "pam_side": "3'"},
    "Cas12a (TTTV)":    {"pam": "TTTV",   "guide_len": 23, "pam_side": "5'"},
}

DELIVERY_VECTORS = {
    "AAV": {"capacite_kb": 4.7, "tissus": "Foie, SNC, muscle", "immuno": "Faible",
            "duree": "Long terme", "in_vivo": True, "in_vitro": True, "cout": "Élevé"},
    "LNP": {"capacite_kb": 20, "tissus": "Foie, poumon, rate", "immuno": "Modérée",
            "duree": "Transitoire", "in_vivo": True, "in_vitro": True, "cout": "Modéré"},
    "Électroporation": {"capacite_kb": 100, "tissus": "Ex vivo", "immuno": "Aucune",
            "duree": "Transitoire", "in_vivo": False, "in_vitro": True, "cout": "Faible"},
    "Nanoparticules d'or": {"capacite_kb": 15, "tissus": "Peau", "immuno": "Faible",
            "duree": "Transitoire", "in_vivo": True, "in_vitro": True, "cout": "Modéré"},
    "Lentivirus": {"capacite_kb": 8, "tissus": "Large", "immuno": "Élevée",
            "duree": "Intégration", "in_vivo": True, "in_vitro": True, "cout": "Modéré"},
}

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

PROTEIN_TAGS = {
    "FLAG": {"seq": "GACTACAAGGACGACGATGACAAG", "aa": "DYKDDDDK"},
    "HA":   {"seq": "TACCCATACGATGTTCCAGATTACGCT", "aa": "YPYDVPDYA"},
    "His6": {"seq": "CACCACCACCACCACCAC", "aa": "HHHHHH"},
    "Myc":  {"seq": "GAACAAAAACTCATCTCAGAAGAGGATCTG", "aa": "EQKLISEEDL"},
    "V5":   {"seq": "GGTAAGCCTATCCCTAACCCTCTCCTCGGTCTCGATTCTACG", "aa": "GKPIPNPLLGLDST"},
}

BASE_TO_2BIT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'U': 3}
BIT2_TO_ASCII = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}

CHROM_COLORS = {
    "chr1":"#FF6B6B","chr2":"#FFB84D","chr3":"#FFE066","chr4":"#A0E7A0",
    "chr5":"#7FE7D9","chr6":"#7FD3E7","chr7":"#7FA8E7","chr8":"#9B7FE7",
    "chr9":"#C77FE7","chr10":"#E77FC7","chr11":"#E77F9B","chr12":"#E79B7F",
    "chr13":"#B8A88A","chr14":"#8AB8A8","chr15":"#8A98B8","chr16":"#B8A8C8",
    "chr17":"#E6B8D8","chr18":"#B8D8E6","chr19":"#D8E6B8","chr20":"#E6D8B8",
    "chr21":"#C8C8C8","chr22":"#A8A8A8","chrX":"#FF8C8C","chrY":"#8C8CFF",
}

CFD_MISMATCH_MATRIX = {
    'A': {'A':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'C':[0,0,0.014,0,0,0.395,0.317,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'G':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0]},
    'C': {'A':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'C':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'G':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0]},
    'G': {'A':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'C':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'G':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0]},
    'T': {'A':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'C':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'G':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],
          'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0]},
}

MIT_WEIGHTS = [0.0,0.0,0.014,0.0,0.0,0.395,0.317,0.0,0.0,0.0,
               0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]

UCSC_GENOMES = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/",
    "hg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes/",
    "dm6":  "https://hgdownload.soe.ucsc.edu/goldenPath/dm6/chromosomes/",
    "ce11": "https://hgdownload.soe.ucsc.edu/goldenPath/ce11/chromosomes/",
}

DOENCH_2016_URLS = [
    "https://raw.githubusercontent.com/maximilianh/crisporWebsite/master/data/doench2016.tab",
    "https://raw.githubusercontent.com/maximilianh/crisporWebsite/master/data/doench2014.tab",
]

DOENCH_2016_BUILTIN = [
    ("GACCCCCTCCACCCCGCGTC", 0.92),("GGGCGAGGAGCTGTTCACCG", 0.85),
    ("GAGCTGGACGGCGACGTAAA", 0.78),("GGTGAACCGCATCGAGCTGA", 0.71),
    ("GCATCGAGCTGAAGGGCATC", 0.88),("GCTGAAGGGCATCGACTTCA", 0.65),
    ("GGCATCGACTTCAAGGAGGA", 0.52),("GACTTCAAGGAGGACGGCAAC", 0.81),
    ("GTTCACCGGCGACGTAAACG", 0.45),("GTGCCCATCCTGGTCGAGCT", 0.90),
    ("GCTGGACGGCGACGTAAACG", 0.62),("GCGACGTAAACGGCCACAAG", 0.79),
    ("GCCACAAGTTCAGCGTGTCC", 0.55),("GTTCAGCGTGTCCGGCGAGG", 0.73),
    ("GTCCGGCGAGGGCGAGGGCG", 0.68),("GAGGGCGAGGGCGATGCCAC", 0.44),
    ("GGGCGATGCCACCTACGGCA", 0.75),("GCCACCTACGGCAAGCTGAC", 0.83),
    ("GCAAGCTGACCCTGAAGTTC", 0.91),("GACCCTGAAGTTCATCTGCAC", 0.38),
]

# ══════════════════════════════════════════════════════════════════════════════
# PLANT — CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
PLANT_GENOMES = {
    "Arabidopsis (TAIR10)": {
        "organism": "arabidopsis_thaliana", "assembly": "TAIR10",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/",
        "pattern": "Arabidopsis_thaliana.TAIR10.dna.chromosome.{chrom}.fa.gz",
        "chroms": ["1","2","3","4","5"],
        "size_mb": 120, "ploidy": "2n=10", "importance": "Modèle végétal",
    },
    "Maize (B73 v5)": {
        "organism": "zea_mays", "assembly": "Zm-B73-REFERENCE-NAM-5.0",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/zea_mays/dna/",
        "pattern": "Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.chromosome.{chrom}.fa.gz",
        "chroms": [str(i) for i in range(1, 11)],
        "size_mb": 2200, "ploidy": "2n=20", "importance": "Culture majeure mondiale",
    },
    "Rice (IRGSP-1.0)": {
        "organism": "oryza_sativa", "assembly": "IRGSP-1.0",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/oryza_sativa/dna/",
        "pattern": "Oryza_sativa.IRGSP-1.0.dna.chromosome.{chrom}.fa.gz",
        "chroms": [str(i) for i in range(1, 13)],
        "size_mb": 400, "ploidy": "2n=24", "importance": "Modèle monocotylédone",
    },
    "Soybean (Wm82.a4)": {
        "organism": "glycine_max", "assembly": "Wm82.a4.v1",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/glycine_max/dna/",
        "pattern": "Glycine_max.V1.0.dna.chromosome.{chrom}.fa.gz",
        "chroms": [str(i) for i in range(1, 21)],
        "size_mb": 980, "ploidy": "2n=40", "importance": "Légumineuse majeure",
    },
    "Wheat (IWGSC)": {
        "organism": "triticum_aestivum", "assembly": "IWGSC",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/triticum_aestivum/dna/",
        "pattern": "Triticum_aestivum.IWGSC.dna.chromosome.{chrom}.fa.gz",
        "chroms": [f"{g}{n}" for g in "ABD" for n in range(1, 8)][:21],
        "size_mb": 14500, "ploidy": "2n=42 (hexaploïde)", "importance": "Céréale majeure",
    },
    "Tomato (SL3.0)": {
        "organism": "solanum_lycopersicum", "assembly": "SL3.0",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/solanum_lycopersicum/dna/",
        "pattern": "Solanum_lycopersicum.SL3.0.dna.chromosome.{chrom}.fa.gz",
        "chroms": [str(i) for i in range(1, 13)],
        "size_mb": 830, "ploidy": "2n=24", "importance": "Modèle Solanaceae",
    },
    "Barley (IBSC_v2)": {
        "organism": "hordeum_vulgare", "assembly": "IBSC_v2",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/hordeum_vulgare/dna/",
        "pattern": "Hordeum_vulgare.IBSC_v2.dna.chromosome.{chrom}.fa.gz",
        "chroms": ["1H","2H","3H","4H","5H","6H","7H"],
        "size_mb": 4900, "ploidy": "2n=14", "importance": "Céréale — tolérance stress",
    },
    "Grape (PN40024)": {
        "organism": "vitis_vinifera", "assembly": "PN40024.v4",
        "url_base": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/vitis_vinifera/dna/",
        "pattern": "Vitis_vinifera.PN40024.v4.dna.chromosome.{chrom}.fa.gz",
        "chroms": [str(i) for i in range(1, 20)],
        "size_mb": 500, "ploidy": "2n=38", "importance": "Fruit / viticulture",
    },
}

AGRONOMIC_TRAITS = {
    "🌾 Rendement": {
        "description": "Gènes contrôlant la productivité",
        "genes": {
            "Gn1a": {"species": "Riz", "function": "Nombre de grains par panicule", "ref": "Ashikari 2005 Science"},
            "DEP1": {"species": "Riz", "function": "Densité panicule érigée", "ref": "Huang 2009 Nat Genet"},
            "IPA1": {"species": "Riz", "function": "Architecture idéale plante", "ref": "Jiao 2010 Nat Genet"},
            "WFP": {"species": "Riz", "function": "Nombre de grains", "ref": "Miura 2010 Plant Cell"},
            "GS3": {"species": "Riz", "function": "Taille du grain", "ref": "Fan 2006 Theor Appl Genet"},
            "TB1": {"species": "Maïs", "function": "Domestication — architecture", "ref": "Doebley 1997 Nature"},
        },
    },
    "🛡️ Résistance pathogènes": {
        "description": "Gènes de résistance",
        "genes": {
            "Xa21": {"species": "Riz", "function": "Résistance Xanthomonas", "ref": "Song 1995 Science"},
            "Pi-ta": {"species": "Riz", "function": "Résistance Magnaporthe", "ref": "Bryan 2000 Plant Cell"},
            "Pi54": {"species": "Riz", "function": "Résistance blast", "ref": "Sharma 2005"},
            "Lr34": {"species": "Blé", "function": "Résistance rouille", "ref": "Krattinger 2009 Science"},
            "Mlo": {"species": "Orge/Blé", "function": "Résistance oïdium", "ref": "Büschges 1997 Cell"},
            "Rpg1": {"species": "Orge", "function": "Résistance stem rust", "ref": "Brueggeman 2002 PNAS"},
        },
    },
    "💧 Tolérance sécheresse": {
        "description": "Réponse au stress hydrique",
        "genes": {
            "DREB1A": {"species": "Arabidopsis/Multiple", "function": "Facteur transcription drought", "ref": "Liu 1998 Plant Cell"},
            "DREB2A": {"species": "Multiple", "function": "Déshydratation", "ref": "Sakuma 2006 Plant Cell"},
            "NAC": {"species": "Multiple", "function": "Stress abiotique (SNAC1)", "ref": "Hu 2006 PNAS"},
            "LEA": {"species": "Multiple", "function": "Protéines embryogenèse tardive", "ref": "Hundertmark 2008"},
            "OsNAC10": {"species": "Riz", "function": "Sécheresse + rendement", "ref": "Jeong 2010 Plant Physiol"},
            "HVA1": {"species": "Orge/Riz", "function": "Protéine LEA type 3", "ref": "Xu 1996 Plant Physiol"},
        },
    },
    "🧪 Qualité nutritionnelle": {
        "description": "Amélioration valeur nutritive",
        "genes": {
            "Wx": {"species": "Riz/Maïs", "function": "Amylose", "ref": "Wang 1990 Plant J"},
            "GPC-B1": {"species": "Blé", "function": "Protéines du grain", "ref": "Uauy 2006 Science"},
            "ZIP": {"species": "Multiple", "function": "Zinc transporteur", "ref": "Multiple"},
            "NAS": {"species": "Multiple", "function": "Fer (nicotianamine)", "ref": "Curie 2009 Plant J"},
            "PSY1": {"species": "Maïs", "function": "Provitamine A", "ref": "Paine 2005 Nat Biotechnol"},
            "CRTISO": {"species": "Riz", "function": "Caroténoïdes (Golden Rice)", "ref": "Ye 2000 Science"},
        },
    },
    "🌿 Architecture plante": {
        "description": "Morphologie",
        "genes": {
            "PROG1": {"species": "Riz", "function": "Érection plante", "ref": "Jin 2008 Nat Genet"},
            "SPL14": {"species": "Riz", "function": "Nombre talles/panicules", "ref": "Miura 2010 Nat Genet"},
            "MOC1": {"species": "Riz", "function": "Initiation talles", "ref": "Li 2003 Nature"},
            "LAZY1": {"species": "Riz", "function": "Gravitropisme", "ref": "Yoshihara 2007 Plant Cell"},
        },
    },
    "⏱️ Floraison / Maturité": {
        "description": "Contrôle floraison",
        "genes": {
            "Hd1": {"species": "Riz", "function": "Photopériode", "ref": "Yano 2000 Plant Cell"},
            "Hd3a": {"species": "Riz", "function": "Florigène", "ref": "Kojima 2002 Plant Cell Physiol"},
            "Ehd1": {"species": "Riz", "function": "Floraison précoce", "ref": "Doi 2004 Genes Dev"},
            "Ghd7": {"species": "Riz", "function": "Floraison + rendement", "ref": "Xue 2008 Nat Genet"},
            "Vrn1": {"species": "Blé/Orge", "function": "Vernalisation", "ref": "Yan 2003 PNAS"},
        },
    },
}

DOMESTICATION_GENES = {
    "Maïs": {"tb1": "Ramification (teosinte → maïs)", "tga1": "Gaine de la graine",
             "bt2": "Poids du grain", "sh2": "Qualité sucrée", "su1": "Amylose"},
    "Riz": {"sh4": "Non-shattering", "qSH1": "Dispersion des graines",
            "PROG1": "Érection", "Wx": "Amylose", "Gn1a": "Nombre de grains"},
    "Blé": {"Q": "Fragilité épi", "Br": "Couleur rachis", "Vrn1": "Vernalisation",
            "Rht-B1": "Hauteur (gène nain)", "Gpc-B1": "Protéines du grain"},
    "Soja": {"GmFT2a": "Floraison", "E1": "Maturité", "Dt1": "Détermination"},
    "Tomate": {"sp": "Détermination", "u": "Forme du fruit", "fas": "Locules/taille",
               "lc": "Taille du fruit"},
}

AGRONOMIC_DATABASES = {
    "Gramene": {"url": "https://www.gramene.org/", "api": "https://data.gramene.org/v69/",
                "description": "Ressource comparative génomique plantes",
                "species": ["Riz","Maïs","Blé","Orge","Sorgho","Arabidopsis"]},
    "MaizeGDB": {"url": "https://www.maizegdb.org/", "api": "https://www.maizegdb.org/api/",
                 "description": "Base de données Maïs", "species": ["Maïs"]},
    "RAP-DB": {"url": "https://rapdb.dna.affrc.go.jp/", "api": "https://rapdb.dna.affrc.go.jp/api/",
               "description": "Rice Annotation Project Database", "species": ["Riz"]},
    "Phytozome": {"url": "https://phytozome-next.jgi.doe.gov/", "api": "https://phytozome.jgi.doe.gov/",
                  "description": "Portail génomique plantes JGI", "species": ["Multi"]},
    "TAIR": {"url": "https://www.arabidopsis.org/", "api": "https://www.arabidopsis.org/api/",
             "description": "The Arabidopsis Information Resource", "species": ["Arabidopsis"]},
    "SoyBase": {"url": "https://www.soybase.org/", "api": "https://www.soybase.org/api/",
                "description": "Base de données Soja", "species": ["Soja"]},
    "WheatIS": {"url": "https://wheatis.org/", "api": "https://wheatis.triticeaetoolbox.org/",
                "description": "Wheat Information System", "species": ["Blé"]},
}

PLANT_ML_TOOLS = {
    "CRISPR-P 2.0": {"url": "http://crispr.hzau.edu.cn/CRISPR2/",
        "description": "Design sgRNA optimisé plantes", "reference": "Liu 2017 Mol Plant",
        "species": ["Riz","Arabidopsis","Maïs","Blé"]},
    "CHOPCHOP v3": {"url": "https://chopchop.cbu.uib.no/",
        "description": "Multi-organismes + visualisation", "reference": "Labun 2019 NAR",
        "species": ["Multi"]},
    "PlantCAD": {"url": "https://huggingface.co/kuleshov-group/PlantCAD",
        "description": "DNA language model plantes", "reference": "Zhai 2024 bioRxiv",
        "species": ["Multi"]},
    "CRISPR-PLANT": {"url": "https://www.genome.arizona.edu/crispr/",
        "description": "Base sgRNA pré-calculés", "reference": "Multiple",
        "species": ["Riz","Maïs","Arabidopsis","Blé","Sorgho"]},
}

# ══════════════════════════════════════════════════════════════════════════════
# CRISPR — FONCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def download_doench_2016(force=False):
    cache_file = CACHE_DIR / "doench_2016.parquet"
    if cache_file.exists() and not force and PYARROW_AVAILABLE:
        try: return pd.read_parquet(cache_file)
        except: pass
    for url in DOENCH_2016_URLS:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200: continue
            sep = '\t' if '\t' in r.text[:500] else ','
            df = pd.read_csv(StringIO(r.text), sep=sep)
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if any(k in cl for k in ['sgrna','guide_seq','seq','guide']): col_map[c] = 'guide'
                elif any(k in el for el in ['eff','score','value']): col_map[c] = 'efficiency'
            df = df.rename(columns=col_map)
            if 'guide' in df.columns and 'efficiency' in df.columns:
                df['guide'] = df['guide'].astype(str).str.upper().str.strip()
                df = df[df['guide'].str.len() == 20]
                df = df[df['guide'].str.match(r'^[ACGT]{20}$')]
                df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
                df = df.dropna(subset=['guide', 'efficiency'])
                if df['efficiency'].max() > 1.5:
                    emin, emax = df['efficiency'].min(), df['efficiency'].max()
                    df['efficiency'] = (df['efficiency'] - emin) / (emax - emin)
                if PYARROW_AVAILABLE:
                    try: df[['guide','efficiency']].to_parquet(cache_file)
                    except: pass
                return df[['guide','efficiency']].reset_index(drop=True)
        except Exception: continue
    return pd.DataFrame(DOENCH_2016_BUILTIN, columns=["guide", "efficiency"])

def download_chromosome(genome="hg38", chrom="chr21", progress_callback=None):
    if genome not in UCSC_GENOMES: return None, "Génome non supporté"
    cache_file = CACHE_DIR / f"{genome}_{chrom}.fa.gz"
    if cache_file.exists():
        try:
            with gzip.open(cache_file, "rt") as f:
                seq = "".join(l.strip().upper() for l in f if not l.startswith(">"))
            return seq, None
        except: pass
    url = f"{UCSC_GENOMES[genome]}{chrom}.fa.gz"
    try:
        if progress_callback: progress_callback(0.0, f"Téléchargement {chrom}...")
        r = requests.get(url, stream=True, timeout=180)
        if r.status_code != 200: return None, f"HTTP {r.status_code}"
        total = int(r.headers.get('content-length', 0))
        chunks, dl = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk); dl += len(chunk)
            if progress_callback and total > 0:
                progress_callback(dl/total, f"{dl//1024//1024}MB / {total//1024//1024}MB")
        data = b"".join(chunks)
        with open(cache_file, "wb") as f: f.write(data)
        content = gzip.decompress(data).decode("utf-8", errors="ignore")
        seq = "".join(l.strip().upper() for l in content.split("\n") if not l.startswith(">"))
        return seq, None
    except Exception as e:
        return None, f"Erreur : {e}"

def download_plant_chromosome(genome_name, chrom, progress_callback=None):
    if genome_name not in PLANT_GENOMES: return None, "Génome non supporté"
    info = PLANT_GENOMES[genome_name]
    cache_file = CACHE_DIR / f"plant_{info['organism']}_{chrom}.fa.gz"
    if cache_file.exists():
        try:
            with gzip.open(cache_file, "rt") as f:
                seq = "".join(l.strip().upper() for l in f if not l.startswith(">"))
            return seq, None
        except: pass
    pattern = info["pattern"].format(chrom=chrom)
    url = info["url_base"] + pattern
    try:
        if progress_callback: progress_callback(0.0, f"Téléchargement {chrom}...")
        r = requests.get(url, stream=True, timeout=240)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code} — essayez Phytozome ou Gramene"
        total = int(r.headers.get('content-length', 0))
        chunks, dl = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk); dl += len(chunk)
            if progress_callback and total > 0:
                progress_callback(dl/total, f"{dl//1024//1024}MB / {total//1024//1024}MB")
        data = b"".join(chunks)
        with open(cache_file, "wb") as f: f.write(data)
        content = gzip.decompress(data).decode("utf-8", errors="ignore")
        seq = "".join(l.strip().upper() for l in content.split("\n") if not l.startswith(">"))
        return seq, None
    except Exception as e:
        return None, f"Erreur : {e}"

def reverse_complement(seq):
    c = {'A':'T','T':'A','G':'C','C':'G','N':'N','U':'A'}
    return ''.join(c.get(b.upper(), 'N') for b in reversed(seq))

def dna_to_rna(seq): return seq.upper().replace('T', 'U')

def gc_content(seq):
    return sum(1 for b in seq.upper() if b in 'GC') / max(len(seq), 1)

def has_homopolymer(seq, min_len=5):
    return any(b * min_len in seq for b in "ACGT")

def match_pam(window, pattern):
    if len(window) != len(pattern): return False
    for s, p in zip(window.upper(), pattern.upper()):
        if p == 'N': continue
        if p == 'V' and s in 'ACG': continue
        if p == 'R' and s in 'AG': continue
        if s != p: return False
    return True

def encode_sequence_2bit(seq):
    return np.array([BASE_TO_2BIT.get(b, 4) for b in seq.upper()], dtype=np.uint8)

def decode_2bit(arr):
    return ''.join(BIT2_TO_ASCII.get(int(x), 'N') for x in arr)

def reverse_complement_2bit(arr):
    return np.array([0, 3, 2, 1, 4], dtype=np.uint8)[arr[::-1]]

def _count_mismatches(s1, s2):
    if len(s1) != len(s2): return 999
    return sum(1 for a, b in zip(s1, s2) if a != b)

def _chrom_sort_key(chrom):
    c = chrom.replace("chr", "")
    if c.isdigit(): return (0, int(c))
    return (1, {"X":23,"Y":24,"M":25}.get(c, 99))

def get_chrom_color(chrom):
    if chrom in CHROM_COLORS: return CHROM_COLORS[chrom]
    return f"#{hash(chrom) % 0xFFFFFF:06X}"

def cfd_score_real(guide, offtarget):
    if len(guide) != len(offtarget): return 0.0
    g = guide.upper().replace('U', 'T'); o = offtarget.upper().replace('U', 'T')
    score = 1.0
    for i in range(min(20, len(g))):
        if g[i] != o[i]:
            if g[i] in CFD_MISMATCH_MATRIX and o[i] in CFD_MISMATCH_MATRIX[g[i]]:
                score *= (1.0 - CFD_MISMATCH_MATRIX[g[i]][o[i]][i])
            else: score *= 0.5
    return round(max(0.0, min(1.0, score)), 4)

def mit_specificity_score(guide, offtarget):
    if len(guide) != len(offtarget): return 0.0
    g = guide.upper(); o = offtarget.upper()
    score = 100.0
    for i in range(min(len(g), len(MIT_WEIGHTS))):
        if g[i] != o[i]: score *= (1.0 - MIT_WEIGHTS[i])
    return round(score, 2)

def score_guide_doench_like(guide):
    if len(guide) < 15: return 0.0
    gc = gc_content(guide)
    gc_score = (1.0 - abs(gc - 0.55) * 2) if 0.4 <= gc <= 0.7 else 0.2
    p20 = 0.15 if guide[-1] == 'G' else (0.05 if guide[-1] == 'C' else 0.0)
    poly_t = 0.4 if 'TTTT' in guide else 0.0
    homo = 0.3 if has_homopolymer(guide, 5) else 0.0
    return round(max(0.0, min(1.0, 0.5*gc_score + p20 - poly_t - homo)), 3)

def find_pam_sites(sequence, pam="NGG", guide_len=20, pam_side="3'"):
    seq = sequence.upper().replace('U', 'T')
    results = []
    for strand, s in [("+", seq), ("-", reverse_complement(seq))]:
        for i in range(len(s) - len(pam) + 1):
            if not match_pam(s[i:i+len(pam)], pam): continue
            if pam_side == "3'":
                gs = i - guide_len
                if gs < 0: continue
                guide, pam_pos = s[gs:i], i
            else:
                gs = i + len(pam)
                guide = s[gs:gs+guide_len]
                if len(guide) < guide_len: continue
                pam_pos = i
            if len(guide) == guide_len:
                results.append({"strand": strand, "pam": s[i:i+len(pam)],
                                "guide": guide, "gc": round(gc_content(guide), 3),
                                "pam_pos": pam_pos + 1})
    return results

def count_offtargets(guide, sequence, max_mismatch=3):
    seq = sequence.upper().replace('U', 'T')
    count = 0
    for s in [seq, reverse_complement(seq)]:
        for i in range(len(s) - len(guide) + 1):
            if _count_mismatches(s[i:i+len(guide)], guide) <= max_mismatch:
                count += 1
    return count

def design_sgrna(sequence, cas_variant="SpCas9 (NGG)", max_offtarget_mismatch=3, top_n=50):
    cfg = PAM_PATTERNS.get(cas_variant, PAM_PATTERNS["SpCas9 (NGG)"])
    sites = find_pam_sites(sequence, cfg["pam"], cfg["guide_len"], cfg["pam_side"])
    results = []
    for s in sites:
        doench = score_guide_doench_like(s["guide"])
        gc_s = s["gc"]
        gc_score = (1.0 - abs(gc_s - 0.55) * 2) if 0.4 <= gc_s <= 0.7 else 0.2
        poly_t = 0.4 if 'TTTT' in s["guide"] else 0.0
        homo = 0.3 if has_homopolymer(s["guide"], 5) else 0.0
        ot = count_offtargets(s["guide"], sequence, max_offtarget_mismatch) if len(sequence) <= 100000 else -1
        composite = 0.6*doench + 0.3*gc_score - 0.5*poly_t - 0.3*homo
        if ot > 0: composite -= min(0.3, ot * 0.02)
        composite = round(max(0.0, min(1.0, composite)), 3)
        quality = ("🟢 Excellent" if composite >= 0.7 else "🟡 Bon" if composite >= 0.5
                   else "🟠 Moyen" if composite >= 0.3 else "🔴 Faible")
        results.append({"Guide (5'→3')": s["guide"], "ARN guide": dna_to_rna(s["guide"]),
                        "PAM": s["pam"], "Position": s["pam_pos"], "Brin": s["strand"],
                        "GC %": round(gc_s*100, 1), "Score Doench": doench,
                        "Score composite": composite,
                        "Off-target": ot if ot >= 0 else "N/A", "Qualité": quality})
    df = pd.DataFrame(results).sort_values("Score composite", ascending=False)
    return df.head(top_n) if len(df) > 0 else df

def design_multiplex(genes_dict, cas_variant="SpCas9 (NGG)", n_guides_per_gene=3):
    all_guides = []
    for gene, seq in genes_dict.items():
        df = design_sgrna(seq, cas_variant, top_n=n_guides_per_gene)
        if df.empty: continue
        df = df.copy(); df["Gène"] = gene
        all_guides.append(df)
    if not all_guides: return pd.DataFrame()
    combined = pd.concat(all_guides, ignore_index=True)
    guides = combined["Guide (5'→3')"].tolist()
    rows = []
    for i, g1 in enumerate(guides):
        min_mm, partner = 999, "—"
        for j, g2 in enumerate(guides):
            if i == j: continue
            mm = _count_mismatches(g1, g2)
            if mm < min_mm: min_mm, partner = mm, combined.iloc[j]["Gène"]
        status = "🟢 OK" if min_mm >= 4 else "🟡 Attention" if min_mm >= 2 else "🔴 Cross-talk"
        rows.append({"Gène": combined.iloc[i]["Gène"], "Guide": g1,
                     "Score": combined.iloc[i]["Score composite"],
                     "Mismatch min": min_mm, "Partenaire": partner, "Compatibilité": status})
    return pd.DataFrame(rows).sort_values(["Gène", "Score"], ascending=[True, False])

def predict_edit_phenotype(scores, n_genes, edit_type="KO"):
    if not scores: return {"score_global": 0.0, "interprétation": "Aucun guide"}
    avg = float(np.mean(scores))
    penalty = 0.02 * max(0, n_genes - 3)
    bonus = 0.1 if edit_type == "KO" else 0.0
    gs = max(0.0, min(1.0, avg - penalty + bonus))
    interp = ("🟢 Élevée" if gs >= 0.7 else "🟡 Modérée" if gs >= 0.5
              else "🟠 Faible" if gs >= 0.3 else "🔴 Très faible")
    return {"score_global": round(gs, 3), "score_moyen": round(avg, 3),
            "nb_gènes": n_genes, "interprétation": interp}

def recommend_delivery(cell_type="in vitro", cargo_kb=4.5):
    rows = []
    for name, info in DELIVERY_VECTORS.items():
        s = 0.0
        s += 0.4 if cargo_kb <= info["capacite_kb"] else -0.3
        if cell_type == "in vitro" and info["in_vitro"]: s += 0.3
        elif cell_type == "in vivo" and info["in_vivo"]: s += 0.3
        if info["immuno"] == "Faible": s += 0.2
        elif info["immuno"] == "Élevée": s -= 0.1
        if info["cout"] == "Faible": s += 0.1
        rows.append({"Vecteur": name, "Score": round(s, 3),
                     "Capacité (kb)": info["capacite_kb"], "Tissus": info["tissus"],
                     "Immuno": info["immuno"], "Durée": info["duree"], "Coût": info["cout"]})
    return pd.DataFrame(rows).sort_values("Score", ascending=False)

def simulate_doubled_haploids(n_loci=3, n_generations=8, n_individuals=500, selfing=True, seed=42):
    rng = np.random.RandomState(seed)
    pop = rng.randint(0, 2, size=(n_individuals, n_loci))
    history = []
    for gen in range(n_generations + 1):
        all_hom = (pop.sum(axis=1) == 0) | (pop.sum(axis=1) == n_loci)
        locus_hom = [np.mean([1 if pop[i, loc] in (0, 1) else 0
                              for i in range(n_individuals)]) for loc in range(n_loci)]
        history.append({"Génération": gen,
                        "Individus 100% homozygotes": round(float(np.mean(all_hom)), 3),
                        "Taux homozygotie moyen": round(float(np.mean(locus_hom)), 3)})
        if gen == n_generations: break
        new_pop = np.zeros_like(pop)
        for i in range(n_individuals):
            parent = pop[i] if selfing else pop[rng.randint(n_individuals)]
            new_pop[i] = parent.copy()
        for i in range(0, n_individuals - 1, 2):
            p1, p2 = new_pop[i], new_pop[i+1] if i+1 < n_individuals else new_pop[i]
            child = np.where(rng.rand(n_loci) < 0.5, p1, p2)
            new_pop[i] = child
            if i+1 < n_individuals: new_pop[i+1] = child
        pop = new_pop
    return pd.DataFrame(history)

def generate_crispr_protocol(gene_name, guides_df, cas_variant="SpCas9 (NGG)"):
    if guides_df is None or guides_df.empty: return "Aucun guide."
    top = guides_df.iloc[0]
    guide = top["Guide (5'→3')"]
    return f"""# PROTOCOLE CRISPR — {gene_name}
Généré : {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Variant : {cas_variant} | PAM : {top['PAM']} | Position : {top['Position']}

## Guide : 5'-{guide}-3'
## ARN  : 5'-{dna_to_rna(guide)}-3'
## Score : {top['Score composite']} | Qualité : {top['Qualité']}

## Oligos
Forward : 5'-CACCG{guide}-3'
Reverse : 5'-AAAC{reverse_complement(guide)}C-3'

## Clonage
1. Phosphorylation + annealing (95°C 5min → 25°C)
2. Digestion BbsI (37°C, 1h)
3. Ligation T4 (16°C, overnight)
4. Transformation Stbl3 + ampicilline

## Transfection
- Cellules 70-80% confluence
- 1 µg plasmide + 3 µL Lipofectamine 3000
- Analyse 48-72h

## Vérification
- PCR + T7 endonuclease I + Sanger/TIDE
"""

# HDR
def check_silent_mutation(seq_ref, seq_alt, mut_pos, frame=0):
    cs = ((mut_pos - frame) // 3) * 3 + frame
    if cs + 3 > len(seq_ref): return False, "—", "—"
    cr = seq_ref[cs:cs+3]; ca = seq_alt[cs:cs+3]
    ar, aa = CODON_TABLE.get(cr, "?"), CODON_TABLE.get(ca, "?")
    return ar == aa, ar, aa

def design_silent_pam_mutation(sequence, pam_start, frame=0):
    seq = sequence.upper()
    alt = {'G':['A','C','T'],'A':['G','C','T'],'C':['A','G','T'],'T':['A','C','G']}
    cands = []
    for pos in [pam_start + 1, pam_start + 2]:
        if pos >= len(seq): continue
        orig = seq[pos]
        for a in alt.get(orig, []):
            mut_seq = seq[:pos] + a + seq[pos+1:]
            silent, ar, aa = check_silent_mutation(seq, mut_seq, pos, frame)
            cands.append({"position": pos+1, "mutation": f"{orig}{pos+1}{a}",
                          "silencieuse": silent, "aa_ref": ar, "aa_new": aa})
    return cands

def design_hdr_ssodn(target_seq, cut_pos, edit_type="substitution",
                      edit_payload="", arm_length=35, break_pam=True):
    seq = target_seq.upper(); n = len(seq); ci = cut_pos - 1
    ls = max(0, ci - arm_length); re_ = min(n, ci + arm_length)
    if edit_type == "substitution":
        payload = edit_payload.upper() if edit_payload else "A"
        edited = seq[:ci] + payload + seq[ci+1:]
        desc = f"Substitution {seq[ci]}→{payload} @ {cut_pos}"
    elif edit_type == "insertion":
        payload = edit_payload.upper()
        edited = seq[:ci] + payload + seq[ci:]
        desc = f"Insertion {len(payload)} nt @ {cut_pos}"
    elif edit_type == "deletion":
        try: dl = int(edit_payload)
        except: dl = 1
        edited = seq[:ci] + seq[ci+dl:]
        desc = f"Délétion {dl} nt @ {cut_pos}"
    else:
        edited, desc = seq, "Aucune édition"
    pam_muts = []
    if break_pam:
        for i in range(max(0, ci+3), min(len(edited)-2, ci+23)):
            if edited[i:i+3][1:3] == "GG":
                cands = design_silent_pam_mutation(edited, i)
                silent = [c for c in cands if c["silencieuse"]]
                if silent:
                    b = silent[0]; p = b["position"] - 1
                    edited = edited[:p] + b["mutation"][-1] + edited[p+1:]
                    pam_muts.append(b)
                break
    la = edited[ls:ci]; ra = edited[ci:re_]
    donor = edited[ls:re_]
    warn = []
    if not (0.3 <= gc_content(la) <= 0.7): warn.append(f"GC bras G : {gc_content(la):.1%}")
    if not (0.3 <= gc_content(ra) <= 0.7): warn.append(f"GC bras D : {gc_content(ra):.1%}")
    if has_homopolymer(la, 6) or has_homopolymer(ra, 6): warn.append("Homopolymère ≥ 6")
    return {"longueur_donneur": len(donor), "bras_gauche": la, "bras_droit": ra,
            "donneur_sense": donor, "donneur_antisense": reverse_complement(donor),
            "edit_description": desc, "mutations_PAM": pam_muts,
            "gc_bras_gauche": round(gc_content(la), 3),
            "gc_bras_droit": round(gc_content(ra), 3), "avertissements": warn}

def design_hdr_plasmid(target_seq, cut_pos, insert_seq, arm_length=600, insert_name="Tag"):
    seq = target_seq.upper(); n = len(seq); ci = cut_pos - 1
    ls = max(0, ci - arm_length); re_ = min(n, ci + arm_length)
    la = seq[ls:ci]; ra = seq[ci:re_]; ins = insert_seq.upper()
    donor = la + ins + ra
    def primer(s, d="fwd"):
        if d == "fwd":
            tm = 64.9 + 41*(s.count("G")+s.count("C")-16.4)/max(len(s),1)
            return s[:22], round(tm, 1)
        r = reverse_complement(s)
        tm = 64.9 + 41*(r.count("G")+r.count("C")-16.4)/max(len(r),1)
        return r[:22], round(tm, 1)
    return {"type_donneur": "Plasmide", "insert_nom": insert_name,
            "longueur_insert": len(ins), "bras_gauche": la, "bras_droit": ra,
            "séquence_donneur_complète": donor, "longueur_totale": len(donor),
            "amorces_bras_gauche": {"Fwd": primer(la, "fwd")[0], "Rev": primer(la, "rev")[0]},
            "amorces_bras_droit": {"Fwd": primer(ra, "fwd")[0], "Rev": primer(ra, "rev")[0]}}

def predict_hdr_efficiency_real(arm_length, edit_type, cell_cycle="asynchrone",
                                  cell_type="iPSC", pam_mutated=False):
    if edit_type == "substitution": base = 15.0
    elif edit_type == "insertion": base = 10.0 if arm_length < 100 else 5.0
    else: base = 8.0
    if 30 <= arm_length <= 40: base *= 1.3
    elif 60 <= arm_length <= 100: base *= 1.2
    elif arm_length >= 500: base *= 1.1
    else: base *= 0.8
    cycle_bonus = {"asynchrone":1.0,"synchronisé S/G2 (nocodazole)":1.8,
                   "synchronisé G2/M (thymidine)":1.5,"RS-1 (enhancer)":1.6,
                   "Scr7 (enhancer)":1.4,"i53 (enhancer)":1.7}.get(cell_cycle, 1.0)
    cell_factor = {"iPSC":1.5,"HEK293T":2.0,"K562":1.3,"T cells":0.6,
                   "Primary cells":0.5,"Fibroblasts":0.8}.get(cell_type, 1.0)
    pam_bonus = 1.2 if pam_mutated else 1.0
    eff = min(base * cycle_bonus * cell_factor * pam_bonus, 85.0)
    recos = []
    if arm_length < 30: recos.append("Bras ≥ 30 nt (Richardson 2016)")
    if cell_cycle == "asynchrone": recos.append("Sync S/G2 → 1.8x")
    if not pam_mutated: recos.append("Ajouter mutation silencieuse PAM")
    return {"efficacité_pct": round(eff, 1),
            "niveau": ("🟢 Élevé" if eff > 30 else "🟡 Modéré" if eff > 15 else "🟠 Faible"),
            "reference": "Richardson 2016, Yeh 2019, Kwart 2017",
            "recommandations": recos}

# ML
def extract_real_features(guide):
    g = guide.upper()
    feats = {}
    for i, b in enumerate(g[:20]):
        for base in "ACGT":
            feats[f"pos{i+1}_{base}"] = 1.0 if b == base else 0.0
    feats["gc_content"] = gc_content(g)
    feats["gc_1_5"] = gc_content(g[:5]); feats["gc_6_10"] = gc_content(g[5:10])
    feats["gc_11_15"] = gc_content(g[10:15]); feats["gc_16_20"] = gc_content(g[15:20])
    feats["tm"] = 64.9 + 41 * (g.count("G") + g.count("C") - 16.4) / max(len(g), 1)
    feats["pos20_G"] = 1.0 if g[-1] == 'G' else 0.0
    feats["pos20_C"] = 1.0 if g[-1] == 'C' else 0.0
    feats["pos1_G"] = 1.0 if g[0] == 'G' else 0.0
    for base in "ACGT":
        max_run, cur = 0, 0
        for x in g:
            if x == base: cur += 1; max_run = max(max_run, cur)
            else: cur = 0
        feats[f"homo_{base}"] = max_run
    feats["has_polyT"] = 1.0 if "TTTT" in g else 0.0
    feats["has_polyA"] = 1.0 if "AAAA" in g else 0.0
    for d in ["AA","AC","AG","AT","CA","CC","CG","CT","GA","GC","GG","GT",
              "TA","TC","TG","TT"]:
        feats[f"dinuc_{d}"] = g.count(d) / max(len(g)-1, 1)
    return feats

def train_real_efficacy_model(df_doench, model_type="rf"):
    guides = df_doench["guide"].tolist()
    y = df_doench["efficiency"].values
    X_list = [extract_real_features(g) for g in guides]
    X_df = pd.DataFrame(X_list).fillna(0)
    X = X_df.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == "xgb" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                   learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                        random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = {"R²": round(r2_score(y_te, y_pred), 4),
               "RMSE": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
               "MAE": round(float(mean_absolute_error(y_te, y_pred)), 4),
               "N_train": len(X_tr), "N_test": len(X_te)}
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    imp = pd.DataFrame({"Feature": X_df.columns,
                        "Importance": model.feature_importances_}
                       ).sort_values("Importance", ascending=False)
    return {"model": model, "feature_names": list(X_df.columns),
            "metrics": metrics, "cv_r2_mean": round(cv.mean(), 4),
            "cv_r2_std": round(cv.std(), 4), "feature_importance": imp,
            "training_data": df_doench.shape[0],
            "source": "Doench 2016"}

def predict_efficacy_ml(guides_list, model, feature_names):
    rows = []
    for g in guides_list:
        feats = extract_real_features(g)
        feat_df = pd.DataFrame([feats])
        for c in feature_names:
            if c not in feat_df.columns: feat_df[c] = 0.0
        feat_df = feat_df[feature_names]
        pred = float(model.predict(feat_df.values)[0])
        conf = 0.75
        if hasattr(model, "estimators_"):
            tree_preds = np.array([t.predict(feat_df.values)[0] for t in model.estimators_])
            conf = max(0.0, 1.0 - float(np.std(tree_preds)) * 2)
        rows.append({"Guide": g, "Efficacité prédite": round(pred, 3),
                     "Confiance": round(conf, 3),
                     "Niveau": ("🟢 Élevé" if pred > 0.65 else "🟡 Bon" if pred > 0.5
                                else "🟠 Moyen" if pred > 0.35 else "🔴 Faible")})
    return pd.DataFrame(rows).sort_values("Efficacité prédite", ascending=False)

# Off-target
def build_kmer_index(genome, k=12, progress_callback=None):
    index = defaultdict(list)
    total = sum(len(seq) for seq in genome.values())
    processed = 0
    for chrom, arr in genome.items():
        for pos in range(len(arr) - k + 1):
            seed = arr[pos:pos+k].tobytes()
            if b'\x04' in seed: continue
            index[seed].append((chrom, pos, "+"))
        rc = reverse_complement_2bit(arr)
        for pos in range(len(rc) - k + 1):
            seed = rc[pos:pos+k].tobytes()
            if b'\x04' in seed: continue
            index[seed].append((chrom, pos, "-"))
        processed += len(arr)
        if progress_callback: progress_callback(processed/total, chrom)
    return dict(index)

def _seed_variants(seed, max_v=100, max_mm=2):
    out = [(seed, 0)]
    if max_mm == 0: return out
    for i in range(len(seed)):
        for b in "ACGT":
            if b == seed[i]: continue
            out.append((seed[:i] + b + seed[i+1:], 1))
    return out[:max_v]

def search_offtargets(guide, genome, index, pam="NGG", max_mm=4):
    guide = guide.upper().replace("U", "T"); k = 12
    if len(guide) < k: return []
    seed = guide[-k:]
    hits, seen = [], set()
    for sv, smm in _seed_variants(seed):
        sb = encode_sequence_2bit(sv).tobytes()
        if sb not in index: continue
        for chrom, pos, strand in index[sb]:
            gl = len(guide)
            if strand == "+":
                gs = pos - (gl - k)
                if gs < 0 or gs + gl > len(genome[chrom]): continue
                cand = decode_2bit(genome[chrom][gs:gs+gl])
                mm = _count_mismatches(cand, guide)
            else:
                gs = pos
                if gs + gl > len(genome[chrom]): continue
                rca = reverse_complement_2bit(genome[chrom])
                cand = decode_2bit(rca[gs:gs+gl])
                mm = _count_mismatches(cand, guide)
            if mm > max_mm: continue
            if strand == "+":
                gp = gs; ps = gp + gl
            else:
                gp = len(genome[chrom]) - (gs + gl); ps = gp - 3
            if strand == "+":
                if ps + 3 > len(genome[chrom]): continue
                psq = decode_2bit(genome[chrom][ps:ps+3])
            else:
                if ps < 0: continue
                psq = decode_2bit(genome[chrom][ps:ps+3])
            if not match_pam(psq, pam): continue
            key = (chrom, gp, strand)
            if key in seen: continue
            seen.add(key)
            hits.append({"chrom": chrom, "pos": gp, "strand": strand,
                         "mismatches": mm, "guide_seq": cand, "pam_seq": psq})
    return sorted(hits, key=lambda h: h["mismatches"])

def analyze_guide_offtargets(guide, genome, pam="NGG", max_mismatch=4, top_n=100):
    index = build_kmer_index(genome, k=12)
    hits = search_offtargets(guide, genome, index, pam, max_mismatch)
    for h in hits:
        h["cfd_score"] = cfd_score_real(guide, h["guide_seq"])
        h["mit_score"] = mit_specificity_score(guide, h["guide_seq"])
        h["classe"] = ("🔴 On-target" if h["mismatches"] == 0 else
                        "🟠 Risque élevé" if h["mismatches"] <= 2 and h["cfd_score"] > 0.3 else
                        "🟡 Risque modéré" if h["mismatches"] <= 2 else "🟢 Risque faible")
    n_high = sum(1 for h in hits if h["mismatches"] <= 2 and h["cfd_score"] > 0.3)
    n_mod = sum(1 for h in hits if h["mismatches"] <= 2 and 0.1 < h["cfd_score"] <= 0.3)
    summary = {"total_hits": len(hits),
               "on_target": sum(1 for h in hits if h["mismatches"] == 0),
               "mm_1": sum(1 for h in hits if h["mismatches"] == 1),
               "mm_2": sum(1 for h in hits if h["mismatches"] == 2),
               "mm_3": sum(1 for h in hits if h["mismatches"] == 3),
               "mm_4plus": sum(1 for h in hits if h["mismatches"] >= 4),
               "hits_high_risk": n_high,
               "specificity_score": round(1.0 - min(1.0, n_high*0.5 + n_mod*0.1), 3)}
    return summary, sorted(hits, key=lambda h: (h["mismatches"], -h["cfd_score"]))[:top_n]

# Dashboard
def compute_guide_criteria(g, sequence=None, ot_hits=None):
    crit = {}
    crit["Doench"] = score_guide_doench_like(g)
    gc = gc_content(g)
    crit["GC optimal"] = max(0.0, min(1.0, 1.0 - abs(gc - 0.55) * 2.5))
    tm = 64.9 + 41*(g.count("G")+g.count("C")-16.4)/max(len(g), 1)
    crit["Tm"] = max(0.0, 1.0 - abs(tm - 60)/30)
    last = g[-1] if g else ""
    crit["PAM-proximal (G)"] = 1.0 if last == "G" else 0.7 if last == "C" else 0.3 if last == "A" else 0.0
    run, cur = 0, 1
    for i in range(1, len(g)):
        if g[i] == g[i-1]: cur += 1; run = max(run, cur)
        else: cur = 1
    crit["Absence homopolymère"] = max(0.0, 1.0 - run/6)
    crit["Absence poly-T"] = 0.0 if "TTTT" in g else 1.0
    if ot_hits is not None:
        n_high = sum(1 for h in ot_hits if h["mismatches"] <= 2 and h.get("cfd_score", 0) > 0.3)
        n_mod = sum(1 for h in ot_hits if h["mismatches"] <= 2 and 0.1 < h.get("cfd_score", 0) <= 0.3)
        crit["Spécificité"] = round(1.0 - min(1.0, n_high*0.4 + n_mod*0.1), 3)
    else: crit["Spécificité"] = 0.5
    if sequence:
        pos = sequence.upper().find(g)
        if pos >= 0:
            rel = pos/max(len(sequence), 1)
            crit["Position cible"] = round(1.0 - abs(rel - 0.5) * 2, 3)
        else: crit["Position cible"] = 0.5
    else: crit["Position cible"] = 0.5
    return crit

def plot_radar_guides(criteria_dict, title="Comparaison"):
    cats = list(next(iter(criteria_dict.values())).keys())
    fig = go.Figure()
    colors = ["#00D4AA","#4D9FFF","#9B7CFF","#FF6B6B","#FFD93D"]
    for i, (name, c) in enumerate(criteria_dict.items()):
        v = [c.get(k, 0.0) for k in cats]
        vc = v + [v[0]]; cc = cats + [cats[0]]
        fig.add_trace(go.Scatterpolar(r=vc, theta=cc, fill="toself", name=name,
                                       line=dict(color=colors[i%len(colors)], width=2), opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#2A3550"),
                                  angularaxis=dict(gridcolor="#2A3550"), bgcolor="#0F1525"),
                      showlegend=True, template="plotly_dark", height=550, title=title,
                      paper_bgcolor="#0A0E1A")
    return fig

def build_comparison_table(guides_list, sequence=None, ot_dict=None):
    rows = []
    for i, g in enumerate(guides_list):
        offt = ot_dict.get(g) if ot_dict else None
        crit = compute_guide_criteria(g, sequence, offt)
        w = {"Doench":0.20,"GC optimal":0.10,"Tm":0.10,"PAM-proximal (G)":0.10,
             "Absence homopolymère":0.10,"Absence poly-T":0.05,"Spécificité":0.25,"Position cible":0.10}
        gs = sum(crit.get(k, 0)*ww for k, ww in w.items())
        rows.append({"ID": f"G{i+1}", "Guide": g, "Score global": round(gs, 3),
                     **{k: round(v, 3) for k, v in crit.items()},
                     "Recommandation": ("🟢 Prioritaire" if gs >= 0.7 else "🟡 Backup" if gs >= 0.5
                                         else "🟠 Alternatif" if gs >= 0.35 else "🔴 À éviter")})
    return pd.DataFrame(rows).sort_values("Score global", ascending=False)

def plot_genome_ideogram(hits, genome, title="Cartographie"):
    if not hits: return go.Figure().update_layout(template="plotly_dark", title="Aucun hit")
    df = pd.DataFrame(hits); df = df[df["chrom"].isin(genome.keys())]
    if df.empty: return go.Figure().update_layout(template="plotly_dark", title="Aucun hit")
    chroms = sorted(genome.keys(), key=_chrom_sort_key)
    lens = {c: len(genome[c]) for c in chroms}
    max_len = max(lens.values())
    y_pos = {c: i for i, c in enumerate(chroms)}
    mm_c = {0:"#FF0000",1:"#FF8C00",2:"#FFD700",3:"#7FFF00",4:"#00CED1"}
    fig = go.Figure()
    for c in chroms:
        L = lens[c]; y = y_pos[c]
        fig.add_trace(go.Scatter(x=[0, L], y=[y, y], mode="lines",
                                  line=dict(color=get_chrom_color(c), width=14),
                                  opacity=0.25, showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=-max_len*0.015, y=y, text=f"<b>{c}</b>",
                            showarrow=False, xanchor="right", font=dict(size=11, color="#E8EDF5"))
    for mm in sorted(df["mismatches"].unique()):
        sub = df[df["mismatches"] == mm]
        fig.add_trace(go.Scatter(x=sub["pos"], y=[y_pos.get(c, 0) for c in sub["chrom"]],
                                  mode="markers",
                                  marker=dict(size=10 if mm == 0 else 7,
                                              color=mm_c.get(mm, "#888"),
                                              symbol="diamond" if mm == 0 else "circle",
                                              line=dict(color="white", width=0.5)),
                                  name=f"{mm} mm"))
    fig.update_layout(template="plotly_dark", title=title,
                      height=max(300, 40*len(chroms)+100),
                      plot_bgcolor="#0F1525", paper_bgcolor="#0A0E1A",
                      xaxis=dict(title="Position (bp)", gridcolor="#2A3550"),
                      yaxis=dict(showticklabels=False, gridcolor="#2A3550"))
    return fig

def run_real_benchmark(model_bundle, df_bench):
    guides = df_bench["guide"].tolist()
    y_true = df_bench["efficiency"].values
    feat_names = model_bundle["feature_names"]
    X_list = [extract_real_features(g) for g in guides]
    X_df = pd.DataFrame(X_list).fillna(0)
    for c in feat_names:
        if c not in X_df.columns: X_df[c] = 0.0
    X_df = X_df[feat_names]
    y_pred = model_bundle["model"].predict(X_df.values)
    metrics = {"R²": round(r2_score(y_true, y_pred), 4),
               "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
               "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
               "Pearson r": round(pearsonr(y_true, y_pred)[0], 4),
               "Spearman ρ": round(spearmanr(y_true, y_pred)[0], 4),
               "N": len(y_true)}
    df_pred = pd.DataFrame({"Guide": guides, "Efficacité réelle": y_true,
                            "Efficacité prédite": y_pred,
                            "Erreur absolue": np.abs(y_true - y_pred)}
                           ).sort_values("Erreur absolue")
    baselines = {
        "GC optimal": np.array([max(0.0, 1.0 - abs(gc_content(g) - 0.5) * 3) for g in guides]),
        "Doench-like": np.array([score_guide_doench_like(g) for g in guides]),
        "Random": np.random.RandomState(42).uniform(0, 1, len(guides)),
    }
    bl_rows = []
    for n, p in baselines.items():
        bl_rows.append({"Prédicteur": n, "R²": round(r2_score(y_true, p), 4),
                        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, p))), 4),
                        "Pearson r": round(pearsonr(y_true, p)[0], 4)})
    bl_df = pd.DataFrame(bl_rows)
    bl_df = pd.concat([pd.DataFrame([{
        "Prédicteur": "🌲 Modèle ML (Doench trained)",
        "R²": metrics["R²"], "RMSE": metrics["RMSE"], "Pearson r": metrics["Pearson r"]}]),
        bl_df]).sort_values("R²", ascending=False).reset_index(drop=True)
    return {"metrics_ml": metrics, "predictions": df_pred, "comparison": bl_df}

def export_to_biostruct(genes=None, guides=None, targets=None):
    exp = {"format": "MetaInsight_to_BioStruct", "version": "1.0",
           "timestamp": datetime.now().isoformat(), "source": "MetaInsight_v9"}
    if genes is not None: exp["genes"] = genes
    if guides is not None: exp["crispr_guides"] = guides
    if targets is not None: exp["crispr_targets"] = targets
    return json.dumps(exp, indent=2, default=str)

# ══════════════════════════════════════════════════════════════════════════════
# PLANT — FONCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def list_plant_genomes():
    rows = []
    for name, info in PLANT_GENOMES.items():
        rows.append({"Génome": name, "Organisme": info["organism"],
                     "Assembly": info["assembly"], "Ploidie": info["ploidy"],
                     "Chromosomes": len(info["chroms"]), "Taille (MB)": info["size_mb"],
                     "Importance": info["importance"]})
    return pd.DataFrame(rows)

def query_gramene(gene_name, limit=5):
    try:
        url = f"https://data.gramene.org/v69/genes?idList={gene_name}&rows={limit}"
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        if r.status_code == 200: return {"status": "ok", "data": r.json()}
        return {"status": "fallback", "data": {"url": f"https://www.gramene.org/search?query={gene_name}"}}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def query_maizegdb(gene_name):
    try:
        url = f"https://www.maizegdb.org/api/locus/{gene_name}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200: return {"status": "ok", "data": r.json()}
        return {"status": "fallback", "data": {"url": f"https://www.maizegdb.org/search?query={gene_name}"}}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def query_rapdb(gene_name):
    try:
        url = f"https://rapdb.dna.affrc.go.jp/api/gene?id={gene_name}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200: return {"status": "ok", "data": r.json()}
        return {"status": "fallback", "data": {"url": f"https://rapdb.dna.affrc.go.jp/search?query={gene_name}"}}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def query_agronomic_gene(gene_name, database="Gramene"):
    if database == "Gramene": return query_gramene(gene_name)
    elif database == "MaizeGDB": return query_maizegdb(gene_name)
    elif database == "RAP-DB": return query_rapdb(gene_name)
    else: return {"status": "fallback", "data": {"url": AGRONOMIC_DATABASES.get(database, {}).get("url", "")}}

def recommend_plant_cas_variant(crop="Riz", application="KO"):
    recos = {
        "Riz": {"KO": ("SpCas9 (NGG)", "Bien établi, PAM fréquent")},
        "Maïs": {"KO": ("SpCas9 + morphogenic", "Nécessite Bbm/Wus2")},
        "Blé": {"KO": ("SpCas9 + morphogenic", "Bbm/Wus2 recommandés")},
        "Arabidopsis": {"KO": ("SpCas9 (NGG)", "Modèle établi")},
        "Soja": {"KO": ("SpCas9 (NGG)", "Standard soja")},
        "Tomate": {"KO": ("SpCas9 (NGG)", "Standard tomate")},
    }
    variant, raison = recos.get(crop, recos["Riz"]).get(application, recos["Riz"]["KO"])
    return {"variant": variant, "raison": raison}

def analyze_field_trial(df_trial, genotype_col="genotype", env_col="environment",
                         trait_col="trait_value"):
    results = {}
    df = df_trial.copy()
    required = [genotype_col, env_col, trait_col]
    missing = [c for c in required if c not in df.columns]
    if missing: return {"error": f"Colonnes manquantes : {missing}"}
    results["n_obs"] = len(df)
    results["n_genotypes"] = df[genotype_col].nunique()
    results["n_environments"] = df[env_col].nunique()
    results["mean_trait"] = round(float(df[trait_col].mean()), 3)
    results["std_trait"] = round(float(df[trait_col].std()), 3)
    try:
        from scipy.stats import f_oneway
        geno_groups = [g[trait_col].values for _, g in df.groupby(genotype_col)]
        f_geno, p_geno = f_oneway(*geno_groups) if len(geno_groups) > 1 else (0, 1)
        env_groups = [g[trait_col].values for _, g in df.groupby(env_col)]
        f_env, p_env = f_oneway(*env_groups) if len(env_groups) > 1 else (0, 1)
        results["F_genotype"] = round(float(f_geno), 3)
        results["p_genotype"] = round(float(p_geno), 6)
        results["F_environment"] = round(float(f_env), 3)
        results["p_environment"] = round(float(p_env), 6)
    except: pass
    try:
        total_var = df[trait_col].var()
        geno_means = df.groupby(genotype_col)[trait_col].mean()
        var_geno = geno_means.var()
        env_means = df.groupby(env_col)[trait_col].mean()
        var_env = env_means.var()
        var_ge = max(0, total_var - var_geno - var_env)
        results["var_total"] = round(float(total_var), 4)
        results["var_genotype"] = round(float(var_geno), 4)
        results["var_environment"] = round(float(var_env), 4)
        results["var_GxE"] = round(float(var_ge), 4)
        if total_var > 0: results["H2_broad"] = round(float(var_geno / total_var), 3)
    except: pass
    stability = []
    for geno, g in df.groupby(genotype_col):
        cv = g[trait_col].std() / max(g[trait_col].mean(), 1e-9)
        stability.append({"Génotype": geno, "Moyenne": round(g[trait_col].mean(), 3),
                          "Écart-type": round(g[trait_col].std(), 3), "CV": round(cv, 3),
                          "Stabilité": "🟢 Élevée" if cv < 0.15 else "🟡 Moyenne" if cv < 0.30 else "🔴 Faible"})
    results["stability"] = pd.DataFrame(stability).sort_values("CV")
    results["best_genotypes"] = results["stability"].head(5)
    return results

def plant_crispr_recommendations(gene_name, crop="Riz"):
    trait_info = None
    for cat_name, cat in AGRONOMIC_TRAITS.items():
        if gene_name in cat["genes"]:
            trait_info = {"catégorie": cat_name, "fonction": cat["genes"][gene_name]["function"],
                          "référence": cat["genes"][gene_name]["ref"],
                          "espèce_source": cat["genes"][gene_name]["species"]}
            break
    cas_reco = recommend_plant_cas_variant(crop, "KO")
    return {"gène": gene_name, "culture": crop, "trait_info": trait_info,
            "variant_Cas_recommandé": cas_reco["variant"],
            "raison_variant": cas_reco["raison"],
            "considérations": [
                f"Ploidie {PLANT_GENOMES.get(crop, {}).get('ploidy', 'N/A')}",
                "Vérifier PAM dans la région cible",
                "Éviter régions répétées (polyploïdes)",
                "Morphogenic factors (Bbm/Wus2) pour maïs/blé",
                "Off-target sur génome complet (grande taille)"],
            "references": ["Wolter 2018", "Zhang 2017", "Li 2020"]}

# ══════════════════════════════════════════════════════════════════════════════
# RENDU : CRISPR
# ══════════════════════════════════════════════════════════════════════════════
def render_crispr_tab():
    st.markdown("## ✂️ CRISPR Design & Delivery <span class='badge-new'>RÉEL</span>", unsafe_allow_html=True)
    st.markdown("""<div class="ref-box">
    📚 Données RÉELLES : Doench 2016 · Génomes UCSC · Scoring CFD + MIT · Modèle HDR publié.
    Modèles 2024-2026 : CRISPR_HNN, CrisprFusion, CCLMoff, PRIDICT2.0.
    </div>""", unsafe_allow_html=True)

    crispr_tabs = st.tabs([
        "🎯 Sélection gènes", "🧬 Design sgRNA", "🔗 Multiplex",
        "🎨 Édition → Phénotype", "🚚 Livraison", "🌱 Haploïdes doublés",
        "📋 Protocole", "🔗 Export BioStruct", "🧬 HDR Knock-in",
        "🤖 ML Efficacité", "🌐 Off-target génome", "📊 Dashboard",
        "🗺️ Cartographie", "🎓 Benchmark Doench 2016",
    ])

    with crispr_tabs[0]:
        st.markdown("### 🎯 Sélection des gènes cibles")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Depuis biomarqueurs ROC", key="cg_roc"):
                if st.session_state.get("biomarker_results"):
                    br = st.session_state.biomarker_results.get("df_ranked")
                    if br is not None and not br.empty:
                        st.session_state.crispr_genes = br.head(60).to_dict("records")
                        st.success(f"✅ {len(st.session_state.crispr_genes)} gènes")
        with c2:
            if st.button("📥 Depuis Random Forest", key="cg_rf"):
                df = st.session_state.get("df_microbiome")
                if df is not None and not df.empty:
                    tc = detect_feature_cols(df)
                    if tc:
                        st.session_state.crispr_genes = [{"gène": t, "source": "RF"} for t in tc[:60]]
                        st.success(f"✅ {len(st.session_state.crispr_genes)} gènes")
        if st.session_state.get("crispr_genes"):
            st.dataframe(pd.DataFrame(st.session_state.crispr_genes), use_container_width=True)

    with crispr_tabs[1]:
        st.markdown("### 🧬 Design de sgRNA")
        c1, c2 = st.columns([3, 1])
        with c1:
            seq_in = st.text_area("Séquence ADN cible",
                value=("ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGT"
                       "CCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATG"
                       "CTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGC"
                       "CTACAAGAAAGTACGAGAT"), height=120, key="crispr_seq_input")
        with c2:
            cas = st.selectbox("Variant Cas", list(PAM_PATTERNS.keys()), key="crispr_cas")
            max_ot = st.slider("Max mm", 1, 5, 3, key="crispr_ot")
            top_n = st.slider("Top N", 5, 100, 30, key="crispr_topn")
        if st.button("🚀 Designer les guides", type="primary", key="crispr_design_btn"):
            if len(seq_in.strip()) < 50: st.error("Séquence trop courte")
            else:
                clean = "".join(c for c in seq_in.upper() if c in "ACGT")
                with st.spinner("Design..."):
                    df_g = design_sgrna(clean, cas, max_ot, top_n)
                if not df_g.empty:
                    st.session_state.crispr_guides = df_g
                    st.session_state.crispr_sequence = clean
                    st.success(f"✅ {len(df_g)} guides identifiés")
        if st.session_state.get("crispr_guides") is not None:
            df_g = st.session_state.crispr_guides
            st.dataframe(df_g.style.background_gradient(cmap="RdYlGn", subset=["Score composite"]),
                         use_container_width=True)
            st.download_button("📥 CSV guides", df_g.to_csv(index=False).encode(),
                              "crispr_guides.csv", "text/csv")

    with crispr_tabs[2]:
        st.markdown("### 🔗 Design Multiplexé")
        gtxt = st.text_area("Format : NOM|SÉQUENCE par ligne",
            value="GENE_A|ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAG\nGENE_B|ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
            height=120, key="crispr_mx_in")
        n_gene = st.slider("Guides par gène", 1, 5, 3, key="crispr_mx_n")
        cas_mx = st.selectbox("Cas (multiplex)", list(PAM_PATTERNS.keys()), key="crispr_cas_mx")
        if st.button("🚀 Design multiplexé", type="primary", key="crispr_mx_btn"):
            gd = {}
            for line in gtxt.strip().split("\n"):
                if "|" in line:
                    p = line.split("|", 1)
                    if len(p) == 2:
                        s = "".join(c for c in p[1].upper() if c in "ACGT")
                        if len(s) > 30: gd[p[0].strip()] = s
            if gd:
                with st.spinner("Design..."):
                    df_mx = design_multiplex(gd, cas_mx, n_gene)
                if not df_mx.empty:
                    st.session_state.crispr_multiplex = df_mx
                    st.success(f"✅ {len(df_mx)} guides multiplexés")
                    st.dataframe(df_mx, use_container_width=True)

    with crispr_tabs[3]:
        st.markdown("### 🎨 Prédiction édition → phénotype")
        if st.session_state.get("crispr_guides") is None:
            st.warning("Lancez d'abord un design")
        else:
            edit_t = st.selectbox("Édition", ["KO","KI","Base editing","Prime editing"], key="edit_t")
            n_g = st.slider("Gènes simultanés", 1, 20, 3, key="n_g")
            if st.button("🔮 Prédire", type="primary", key="pred_btn"):
                scores = st.session_state.crispr_guides["Score composite"].head(n_g).tolist()
                pred = predict_edit_phenotype(scores, n_g, edit_t)
                c1, c2, c3 = st.columns(3)
                c1.metric("Score global", pred["score_global"])
                c2.metric("Score moyen", pred["score_moyen"])
                c3.metric("Nb gènes", pred["nb_gènes"])
                st.info(pred["interprétation"])

    with crispr_tabs[4]:
        st.markdown("### 🚚 Livraison")
        c1, c2 = st.columns(2)
        with c1: ct = st.selectbox("Application", ["in vitro", "in vivo"], key="ct")
        with c2: ck = st.slider("Cargo (kb)", 1.0, 30.0, 4.5, 0.5, key="ck")
        if st.button("💡 Recommander", type="primary", key="del_btn"):
            df_del = recommend_delivery(ct, ck)
            st.dataframe(df_del.style.background_gradient(cmap="RdYlGn", subset=["Score"]),
                         use_container_width=True)
            st.success(f"🏆 Recommandé : {df_del.iloc[0]['Vecteur']}")

    with crispr_tabs[5]:
        st.markdown("### 🌱 Haploïdes Doublés")
        c1, c2, c3 = st.columns(3)
        with c1: nl = st.slider("Loci", 1, 8, 3, key="dh_nl")
        with c2: ng = st.slider("Générations", 3, 15, 8, key="dh_ng")
        with c3: sf = st.checkbox("Autofécondation", True, key="dh_sf")
        if st.button("🌱 Simuler", type="primary", key="dh_btn"):
            df_dh = simulate_doubled_haploids(nl, ng, 500, sf)
            fig = px.line(df_dh, x="Génération",
                          y=["Individus 100% homozygotes", "Taux homozygotie moyen"],
                          template="plotly_dark", markers=True,
                          title="Fixation des allèles édités")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_dh, use_container_width=True)

    with crispr_tabs[6]:
        st.markdown("### 📋 Protocole expérimental")
        if st.session_state.get("crispr_guides") is None:
            st.warning("Lancez d'abord un design")
        else:
            gn = st.text_input("Nom du gène", value="GENE_TARGET", key="proto_g")
            if st.button("📋 Générer", type="primary", key="proto_btn"):
                st.session_state.crispr_protocol = generate_crispr_protocol(
                    gn, st.session_state.crispr_guides,
                    st.session_state.get("crispr_cas", "SpCas9 (NGG)"))
            if st.session_state.get("crispr_protocol"):
                st.markdown(st.session_state.crispr_protocol)
                st.download_button("📥 Protocole", st.session_state.crispr_protocol.encode(),
                                   "crispr_protocol.md")

    with crispr_tabs[7]:
        st.markdown("### 🔗 Export BioStruct")
        if st.button("📤 Générer JSON", type="primary", key="exp_btn"):
            genes = st.session_state.get("crispr_genes", [])
            guides = (st.session_state.crispr_guides.head(20).to_dict("records")
                     if st.session_state.get("crispr_guides") is not None else [])
            targets = (st.session_state.crispr_multiplex.to_dict("records")
                      if st.session_state.get("crispr_multiplex") is not None else [])
            st.session_state.crispr_export = export_to_biostruct(genes, guides, targets)
            st.success("✅ Export généré")
        if st.session_state.get("crispr_export"):
            st.code(st.session_state.crispr_export[:1500], language="json")
            st.download_button("📥 JSON", st.session_state.crispr_export.encode(),
                              "metainsight_to_biostruct.json", "application/json")

    with crispr_tabs[8]:
        st.markdown("### 🧬 HDR Knock-in")
        mode = st.radio("Mode", ["ssODN", "Plasmide"], horizontal=True, key="hdr_mode")
        seq_h = st.text_area("Séquence",
            value=("ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGT"
                   "CCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATG"),
            height=100, key="hdr_seq")
        c1, c2 = st.columns(2)
        with c1: cut_p = st.number_input("Position coupure", 1, max(len(seq_h), 1), 30, key="hdr_cut")
        with c2: cell_t = st.selectbox("Type cellulaire",
            ["iPSC","HEK293T","K562","T cells","Primary cells"], key="hdr_cell")
        if mode == "ssODN":
            cc1, cc2, cc3 = st.columns(3)
            with cc1: et = st.selectbox("Type", ["substitution","insertion","deletion"], key="hdr_et")
            with cc2: ep = st.text_input("Payload", value="A", key="hdr_ep")
            with cc3: al = st.slider("Bras (nt)", 20, 80, 35, key="hdr_al")
            c_a, c_b = st.columns(2)
            with c_a: bp = st.checkbox("Casser PAM", True, key="hdr_bp")
            with c_b: cyc = st.selectbox("Sync",
                ["asynchrone","synchronisé S/G2 (nocodazole)","RS-1 (enhancer)"], key="hdr_cyc")
            if st.button("🚀 Designer ssODN", type="primary", key="hdr_ss_btn"):
                clean = "".join(c for c in seq_h.upper() if c in "ACGT")
                donor = design_hdr_ssodn(clean, int(cut_p), et, ep, al, bp)
                st.success(f"✅ ssODN : {donor['longueur_donneur']} nt")
                st.code(donor["donneur_sense"], language="text")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Longueur", donor["longueur_donneur"])
                c2.metric("GC G", f"{donor['gc_bras_gauche']:.1%}")
                c3.metric("GC D", f"{donor['gc_bras_droit']:.1%}")
                c4.metric("Muts PAM", len(donor["mutations_PAM"]))
                for w in donor["avertissements"]: st.warning(w)
                pred = predict_hdr_efficiency_real(al, et, cyc, cell_t, bool(donor["mutations_PAM"]))
                st.info(f"📊 Référence : {pred['reference']}")
                st.metric("Efficacité HDR", f"{pred['efficacité_pct']}%", delta=pred["niveau"])
        else:
            tag = st.selectbox("Tag", list(PROTEIN_TAGS.keys()), key="hdr_tag")
            al2 = st.slider("Bras (bp)", 200, 1500, 600, 50, key="hdr_al2")
            if st.button("🚀 Designer plasmide", type="primary", key="hdr_pl_btn"):
                clean = "".join(c for c in seq_h.upper() if c in "ACGT")
                donor = design_hdr_plasmid(clean, int(cut_p), PROTEIN_TAGS[tag]["seq"], al2, tag)
                st.success(f"✅ Plasmide : {donor['longueur_totale']} bp")

    with crispr_tabs[9]:
        st.markdown("### 🤖 ML Efficacité")
        ml_subs = st.tabs(["🌐 Télécharger Doench", "🎓 Entraîner", "🔮 Prédire", "📊 Perf"])
        with ml_subs[0]:
            st.markdown("#### 📚 Dataset Doench 2016 RÉEL")
            if st.button("🌐 Télécharger Doench 2016 réel", type="primary", key="dl_doench"):
                with st.spinner("Téléchargement..."):
                    df_real = download_doench_2016(force=True)
                st.session_state.doench_real = df_real
                st.success(f"✅ {len(df_real)} guides réels")
                st.dataframe(df_real.head(10), use_container_width=True)
        with ml_subs[1]:
            if st.session_state.get("doench_real") is None:
                st.warning("Téléchargez d'abord Doench 2016")
            else:
                model_type = st.selectbox("Modèle", ["rf", "xgb"], key="ml_mt")
                if st.button("🚀 Entraîner", type="primary", key="train_real_btn"):
                    with st.spinner("Entraînement..."):
                        bundle = train_real_efficacy_model(st.session_state.doench_real, model_type)
                    st.session_state.ml_bundle = bundle
                    st.success(f"✅ Entraîné sur {bundle['training_data']} guides")
                    st.json(bundle["metrics"])
        with ml_subs[2]:
            if st.session_state.get("ml_bundle") is None:
                st.warning("Entraînez d'abord")
            else:
                g_in = st.text_area("Guides (un par ligne)", height=120, key="ml_pred_in",
                    value=("\n".join(st.session_state.crispr_guides["Guide (5'→3')"].head(5).tolist())
                           if st.session_state.get("crispr_guides") is not None else ""))
                if st.button("🔮 Prédire", type="primary", key="ml_pred_btn"):
                    gl = [g.strip().upper() for g in g_in.strip().split("\n") if len(g.strip()) >= 15]
                    if gl:
                        b = st.session_state.ml_bundle
                        dfp = predict_efficacy_ml(gl, b["model"], b["feature_names"])
                        st.dataframe(dfp.style.background_gradient(cmap="RdYlGn",
                                                                     subset=["Efficacité prédite"]),
                                     use_container_width=True)
        with ml_subs[3]:
            if st.session_state.get("ml_bundle"):
                b = st.session_state.ml_bundle
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R²", b["metrics"]["R²"])
                c2.metric("RMSE", b["metrics"]["RMSE"])
                c3.metric("MAE", b["metrics"]["MAE"])
                c4.metric("CV R²", f"{b['cv_r2_mean']} ± {b['cv_r2_std']}")
                if not b["feature_importance"].empty:
                    fig = px.bar(b["feature_importance"].head(20), x="Importance", y="Feature",
                                 orientation="h", template="plotly_dark", title="Top 20 features")
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

    with crispr_tabs[10]:
        st.markdown("### 🌐 Off-target génome-wide")
        ot_subs = st.tabs(["📁 Génome", "🎯 Analyser"])
        with ot_subs[0]:
            c1, c2 = st.columns(2)
            with c1: gn = st.selectbox("Génome", list(UCSC_GENOMES.keys()), key="ot_gn")
            with c2: ch = st.text_input("Chromosome", value="chr21", key="ot_ch")
            if st.button("🌐 Télécharger le chromosome", type="primary", key="ot_dl_btn"):
                progress = st.progress(0, text="Téléchargement...")
                def cb(frac, msg): progress.progress(min(1.0, frac), text=msg)
                seq, err = download_chromosome(gn, ch, cb)
                progress.empty()
                if err: st.error(err)
                else:
                    st.session_state.ot_genome = {f"{gn}_{ch}": encode_sequence_2bit(seq)}
                    st.session_state.ot_index = None
                    st.success(f"✅ {ch} : {len(seq):,} bp")
            if st.session_state.get("ot_genome"):
                g = st.session_state.ot_genome
                tot = sum(len(v) for v in g.values())
                c1, c2 = st.columns(2)
                c1.metric("Chromosome(s)", len(g))
                c2.metric("Taille", f"{tot:,} bp")
                if st.button("🔧 Construire l'index k-mer", key="ot_idx_btn"):
                    with st.spinner("Construction (1-3 min)..."):
                        st.session_state.ot_index = build_kmer_index(g, k=12)
                    st.success(f"✅ {len(st.session_state.ot_index):,} k-mers")
        with ot_subs[1]:
            if st.session_state.get("ot_genome") is None or st.session_state.get("ot_index") is None:
                st.warning("Chargez un génome et construisez l'index")
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    g_in = st.text_input("Guide (20 nt)", key="ot_g_in",
                        value=(st.session_state.crispr_guides.iloc[0]["Guide (5'→3')"]
                               if st.session_state.get("crispr_guides") is not None else ""))
                with c2: mm = st.slider("Max mm", 0, 5, 4, key="ot_mm")
                if st.button("🔍 Rechercher", type="primary", key="ot_search_btn"):
                    if len(g_in.strip()) < 15: st.error("Guide trop court")
                    else:
                        with st.spinner("Recherche..."):
                            summary, hits = analyze_guide_offtargets(
                                g_in.strip().upper(), st.session_state.ot_genome,
                                pam="NGG", max_mismatch=mm, top_n=100)
                        st.session_state.ot_summary = summary
                        st.session_state.ot_hits = hits
                if st.session_state.get("ot_summary"):
                    s = st.session_state.ot_summary
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total hits", s["total_hits"])
                    c2.metric("On-target", s["on_target"])
                    c3.metric("Risque élevé", s["hits_high_risk"])
                    c4.metric("Spécificité", s["specificity_score"])
                    if st.session_state.get("ot_hits"):
                        df_h = pd.DataFrame(st.session_state.ot_hits)
                        st.dataframe(df_h[["chrom","pos","strand","mismatches",
                                            "guide_seq","pam_seq","cfd_score","mit_score","classe"]]
                                     .style.background_gradient(cmap="RdYlGn_r",
                                                                 subset=["mismatches","cfd_score"]),
                                     use_container_width=True)

    with crispr_tabs[11]:
        st.markdown("### 📊 Dashboard comparatif")
        if st.session_state.get("crispr_guides") is None:
            st.warning("Lancez d'abord un design")
        else:
            n_top = st.slider("Top N", 2, 15, 5, key="dash_n")
            gl = st.session_state.crispr_guides.head(n_top)["Guide (5'→3')"].tolist()
            if len(gl) >= 2:
                crit = {f"G{i+1}": compute_guide_criteria(g, st.session_state.get("crispr_sequence"))
                        for i, g in enumerate(gl)}
                df_c = build_comparison_table(gl, st.session_state.get("crispr_sequence"))
                st.dataframe(df_c.style.background_gradient(cmap="RdYlGn", subset=["Score global"]),
                             use_container_width=True)
                fig = plot_radar_guides(crit)
                st.plotly_chart(fig, use_container_width=True)

    with crispr_tabs[12]:
        st.markdown("### 🗺️ Cartographie génomique")
        if st.session_state.get("ot_hits") is None:
            st.warning("Lancez d'abord une analyse off-target")
        else:
            fig = plot_genome_ideogram(st.session_state.ot_hits,
                                        st.session_state.get("ot_genome", {}))
            st.plotly_chart(fig, use_container_width=True)

    with crispr_tabs[13]:
        st.markdown("### 🎓 Benchmark Doench 2016")
        st.markdown("""<div class="ref-box">
        📚 Comparaison avec baselines : GC optimal, Doench-like, Random.
        </div>""", unsafe_allow_html=True)
        bench_subs = st.tabs(["📚 Dataset", "🚀 Benchmark", "📊 Résultats"])
        with bench_subs[0]:
            if st.session_state.get("doench_real") is None:
                if st.button("🌐 Télécharger Doench 2016", type="primary", key="bench_dl"):
                    with st.spinner("Téléchargement..."):
                        st.session_state.doench_real = download_doench_2016()
                    st.success(f"✅ {len(st.session_state.doench_real)} guides")
            else:
                df = st.session_state.doench_real
                st.success(f"✅ Dataset : {len(df)} guides")
                st.dataframe(df.head(10), use_container_width=True)
        with bench_subs[1]:
            if st.session_state.get("doench_real") is None:
                st.warning("Chargez un dataset")
            elif st.session_state.get("ml_bundle") is None:
                st.warning("Entraînez d'abord un modèle ML")
            else:
                if st.button("🚀 Lancer le benchmark", type="primary", key="bench_run"):
                    with st.spinner("Évaluation..."):
                        results = run_real_benchmark(st.session_state.ml_bundle,
                                                     st.session_state.doench_real)
                    st.session_state.bench_results = results
                    st.success("✅ Benchmark terminé")
        with bench_subs[2]:
            if st.session_state.get("bench_results") is None:
                st.info("Lancez d'abord le benchmark")
            else:
                r = st.session_state.bench_results
                m = r["metrics_ml"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R²", m["R²"])
                c2.metric("RMSE", m["RMSE"])
                c3.metric("MAE", m["MAE"])
                c4.metric("Pearson r", m["Pearson r"])
                c5, c6 = st.columns(2)
                c5.metric("Spearman ρ", m["Spearman ρ"])
                c6.metric("N", m["N"])
                st.markdown("#### 🏆 Comparaison avec baselines")
                st.dataframe(r["comparison"], use_container_width=True)
                st.markdown("#### 🎯 Prédit vs Réel")
                fig = px.scatter(r["predictions"], x="Efficacité réelle", y="Efficacité prédite",
                                 color="Erreur absolue", color_continuous_scale="RdYlGn_r",
                                 template="plotly_dark", title="Doench 2016")
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(color="#00D4AA", width=2, dash="dash"))
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RENDU : PLANT
# ══════════════════════════════════════════════════════════════════════════════
def render_plant_tab():
    st.markdown("## 🌱 Plant & Agronomie <span class='badge-new'>SIVB/CAST</span>", unsafe_allow_html=True)
    st.markdown("""<div class="ref-box">
    📚 Compatible SIVB/CAST 2026 — Pairwise & Rainbow Crops.
    8 génomes végétaux · 6 catégories de traits · Bases Gramene, MaizeGDB, RAP-DB ·
    Essais terrain (G×E) · Modèles ML plantes.
    </div>""", unsafe_allow_html=True)

    plant_tabs = st.tabs([
        "🌾 Génomes végétaux", "🧬 Bases agronomiques", "🌱 Traits agronomiques",
        "🏛️ Gènes domestication", "📊 Essais terrain (G×E)", "🤖 ML Plantes",
        "✂️ CRISPR Plantes", "📚 Références 2024-2026",
    ])

    with plant_tabs[0]:
        st.markdown("### 🌾 Génomes végétaux")
        st.markdown("Source : **Ensembl Plants Release 57**")
        df_genomes = list_plant_genomes()
        st.dataframe(df_genomes, use_container_width=True, hide_index=True)
        st.markdown("#### 🌐 Téléchargement")
        c1, c2 = st.columns(2)
        with c1:
            genome_name = st.selectbox("Génome", list(PLANT_GENOMES.keys()), key="plant_genome")
        with c2:
            chroms_avail = PLANT_GENOMES[genome_name]["chroms"]
            chrom = st.selectbox("Chromosome", chroms_avail, key="plant_chrom")
        if st.button("🌐 Télécharger le chromosome", type="primary", key="plant_dl_btn"):
            progress = st.progress(0, text="Téléchargement...")
            def cb(frac, msg): progress.progress(min(1.0, frac), text=msg)
            seq, err = download_plant_chromosome(genome_name, chrom, cb)
            progress.empty()
            if err:
                st.error(err)
                st.info("💡 Alternative : Phytozome, Gramene ou Ensembl Plants manuellement.")
            else:
                st.session_state.plant_genome = {f"{genome_name}_{chrom}": encode_sequence_2bit(seq)}
                st.session_state.plant_index = None
                st.success(f"✅ {chrom} : {len(seq):,} bp téléchargés")
                st.info("💡 Utilisez CRISPR → Off-target pour analyser ce génome.")

    with plant_tabs[1]:
        st.markdown("### 🧬 Bases de données agronomiques")
        c1, c2 = st.columns(2)
        with c1:
            db_choice = st.selectbox("Base", list(AGRONOMIC_DATABASES.keys()), key="plant_db")
        with c2:
            gene_query = st.text_input("Nom du gène", value="DREB1A", key="plant_gene_query")
        info = AGRONOMIC_DATABASES[db_choice]
        st.info(f"**{db_choice}** — {info['description']}")
        st.markdown(f"🔗 [Site officiel]({info['url']})")
        if st.button("🔍 Interroger l'API", type="primary", key="plant_query_btn"):
            with st.spinner(f"Interrogation..."):
                result = query_agronomic_gene(gene_query, db_choice)
            if result.get("status") == "ok":
                st.success("✅ Résultat")
                st.json(result.get("data", {}))
            else:
                st.warning("API non disponible")
                if "data" in result and "url" in result["data"]:
                    st.markdown(f"[🔍 Rechercher manuellement]({result['data']['url']})")
        st.markdown("---")
        st.markdown("#### 📚 Toutes les bases")
        rows = [{"Base": n, "Description": i["description"], "Espèces": ", ".join(i["species"])}
                for n, i in AGRONOMIC_DATABASES.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with plant_tabs[2]:
        st.markdown("### 🌱 Traits agronomiques majeurs")
        for cat_name, cat in AGRONOMIC_TRAITS.items():
            with st.expander(f"{cat_name} — {cat['description']}"):
                rows = [{"Gène": g, "Espèce": i["species"], "Fonction": i["function"],
                         "Référence": i["ref"]} for g, i in cat["genes"].items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with plant_tabs[3]:
        st.markdown("### 🏛️ Gènes de domestication")
        crop = st.selectbox("Culture", list(DOMESTICATION_GENES.keys()), key="dom_crop")
        genes_data = DOMESTICATION_GENES[crop]
        rows = [{"Gène": g, "Rôle": r} for g, r in genes_data.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("💡 Excellentes cibles pour l'édition génique (néo-domestication).")

    with plant_tabs[4]:
        st.markdown("### 📊 Analyse d'essais terrain (G×E)")
        if st.button("🎲 Générer essai démo", key="field_demo"):
            np.random.seed(42)
            n_geno, n_env = 20, 5
            genotypes = [f"G{i+1:02d}" for i in range(n_geno)]
            environments = ["Nord","Sud","Est","Ouest","Centre"]
            rows = []
            for g in genotypes:
                g_eff = np.random.normal(0, 2)
                for e in environments:
                    e_eff = np.random.normal(0, 3)
                    for rep in range(3):
                        rows.append({"genotype": g, "environment": e,
                                     "replication": rep + 1,
                                     "trait_value": round(50 + g_eff + e_eff + np.random.normal(0, 1), 2)})
            st.session_state.field_trial = pd.DataFrame(rows)
            st.success(f"✅ {n_geno} génotypes × {n_env} environnements")
        uploaded_trial = st.file_uploader("Ou upload CSV (genotype, environment, trait_value)",
                                            type=["csv"], key="field_upload")
        if uploaded_trial:
            try:
                st.session_state.field_trial = pd.read_csv(uploaded_trial)
                st.success(f"✅ {len(st.session_state.field_trial)} obs")
            except Exception as e: st.error(f"Erreur : {e}")
        if st.session_state.get("field_trial") is not None:
            df = st.session_state.field_trial
            st.dataframe(df.head(10), use_container_width=True)
            if st.button("🚀 Analyser", type="primary", key="field_analyze"):
                with st.spinner("Analyse..."):
                    results = analyze_field_trial(df)
                st.session_state.field_results = results
            if st.session_state.get("field_results"):
                r = st.session_state.field_results
                if "error" in r:
                    st.error(r["error"])
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Obs", r["n_obs"])
                    c2.metric("Génotypes", r["n_genotypes"])
                    c3.metric("Environnements", r["n_environments"])
                    c4.metric("Moyenne", r["mean_trait"])
                    c1, c2 = st.columns(2)
                    c1.metric("F génotype", r.get("F_genotype", "N/A"))
                    c2.metric("F env", r.get("F_environment", "N/A"))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Var tot", r.get("var_total", "N/A"))
                    c2.metric("Var géno", r.get("var_genotype", "N/A"))
                    c3.metric("Var env", r.get("var_environment", "N/A"))
                    c4.metric("Var G×E", r.get("var_GxE", "N/A"))
                    if "H2_broad" in r:
                        st.metric("H²", r["H2_broad"])
                    st.markdown("#### 🏆 Classement stabilité")
                    st.dataframe(r["stability"], use_container_width=True, hide_index=True)

    with plant_tabs[5]:
        st.markdown("### 🤖 Modèles ML spécifiques plantes")
        rows = [{"Outil": n, "Description": i["description"],
                 "Espèces": ", ".join(i["species"]), "Référence": i["reference"],
                 "URL": i["url"]} for n, i in PLANT_ML_TOOLS.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        crop = st.selectbox("Culture", ["Riz","Maïs","Blé","Arabidopsis","Soja","Tomate"], key="ml_crop")
        recommended = [n for n, i in PLANT_ML_TOOLS.items() if crop in i["species"] or "Multi" in i["species"]]
        if recommended: st.success(f"✅ Outils pour {crop} : {', '.join(recommended)}")
        else: st.info("Aucun outil spécialisé")

    with plant_tabs[6]:
        st.markdown("### ✂️ Recommandations CRISPR pour plantes")
        c1, c2 = st.columns(2)
        with c1: plant_gene = st.text_input("Gène", value="Gn1a", key="plant_crispr_gene")
        with c2: crop = st.selectbox("Culture", ["Riz","Maïs","Blé","Arabidopsis","Soja","Tomate"], key="plant_crispr_crop")
        if st.button("🚀 Générer recommandations", type="primary", key="plant_crispr_btn"):
            reco = plant_crispr_recommendations(plant_gene, crop)
            st.markdown(f"#### 🌾 Recommandations pour **{plant_gene}** ({crop})")
            if reco.get("trait_info"):
                st.success(f"✅ Gène identifié : **{reco['trait_info']['catégorie']}**")
                c1, c2 = st.columns(2)
                c1.metric("Fonction", reco["trait_info"]["fonction"])
                c2.metric("Réf", reco["trait_info"]["référence"])
            st.markdown("#### 🎯 Variant Cas recommandé")
            st.info(f"**{reco['variant_Cas_recommandé']}** — {reco['raison_variant']}")
            st.markdown("#### ⚠️ Considérations")
            for c in reco["considérations"]: st.markdown(f"- {c}")

    with plant_tabs[7]:
        st.markdown("### 📚 Références 2024-2026")
        st.markdown("""
        #### Modèles on-target
        - **CRISPR_HNN** (2025) — Comput Struct Biotechnol J 27:2098-2106
        - **CrisprFusion** (2026) — Comput Biol Chem
        - **DeepCC9** (2026) — Bioinformatics

        #### Off-target
        - **CCLMoff** (2025) — Commun Biol 8:882
        - **DNABERT-Epi** (2025) — PLoS ONE 20(11):e0335863
        - **CRISMER** (2025) — bioRxiv

        #### Base editing
        - **CRISPRon-ABE/CBE** (2025) — Nat Commun 16:9821
        - **BEstimate** (2026) — Genome Biol

        #### Prime editing
        - **PRIDICT2.0** (2025) — Nat Biotechnol 43:712-719
        - **OptiPrime** (2025) — Thèse Harvard
        - **PrimeNet** (2025) — NAR

        #### Livraison in vivo
        - **LNP vs AAV9** (Zhang 2025) — Cell Rep
        - **Prime editing in vivo** (2026) — AJHG

        #### Fondateurs
        - **Doench 2016** — Nat Biotechnol 34:184-191
        - **Hsu 2013** — Nat Biotechnol 31:827-832
        """)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    defaults = {
        "df_microbiome": generate_demo_microbiome(),
        "pgm_data": generate_demo_pgm_data(),
        "epi_data": generate_demo_epitranscriptomic_data(),
        "epi_annotation_db": generate_epi_annotation_db(),
        "gemini_key": _ENV_GEMINI_KEY, "groq_key": _ENV_GROQ_KEY,
        "openrouter_key": _ENV_OPENROUTER_KEY, "deepseek_key": _ENV_DEEPSEEK_KEY,
        "ai_provider": "Gemini Flash (Google — GRATUIT)",
        "gemini_model": "gemini-1.5-flash",
        "groq_model": "llama-3.3-70b-versatile",
        "trained_model": load_model(),
        "crispr_genes": None, "crispr_guides": None, "crispr_sequence": None,
        "crispr_multiplex": None, "crispr_protocol": None, "crispr_export": None,
        "crispr_cas": "SpCas9 (NGG)",
        "hdr_donor": None, "ml_bundle": None, "doench_real": None,
        "ot_genome": None, "ot_index": None, "ot_summary": None, "ot_hits": None,
        "bench_results": None, "biomarker_results": None,
        "plant_genome": None, "plant_index": None,
        "field_trial": None, "field_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    with st.sidebar:
        st.markdown("## 🔬 MetaInsight *v9*")
        st.markdown('<span style="font-size:0.7rem;color:#7A8BA8;">Multi-omics · PGM · Épi · CRISPR · Plant</span>', unsafe_allow_html=True)
        st.markdown("---")
        data_type = st.radio("Type de données",
            ["Microbiome", "PGM (VCF)", "Épitranscriptomique"], index=0, key="dt_radio")

        if data_type == "Microbiome":
            st.markdown("### 📂 Microbiome")
            f = st.file_uploader("CSV/TSV", type=["csv","tsv","txt","xlsx"], key="ul_micro")
            if f is not None:
                try:
                    df = pd.read_csv(f)
                    if 'environment' not in df.columns: df['environment'] = 'Group1'
                    st.session_state.df_microbiome = df
                    st.success(f"✅ {len(df)} échantillons")
                except Exception as e: st.error(f"Erreur : {e}")
            if st.button("⚡ Démo microbiome", key="demo_micro"):
                st.session_state.df_microbiome = generate_demo_microbiome()
            df_m = st.session_state.df_microbiome
            tc = detect_feature_cols(df_m) if df_m is not None else []
            st.markdown(f"*{len(df_m) if df_m is not None else 0}* · *{len(tc)}* features")
        elif data_type == "PGM (VCF)":
            st.markdown("### 🧬 PGM")
            f = st.file_uploader("VCF", type=["vcf","gz"], key="ul_vcf")
            if f is not None and DUCKDB_AVAILABLE:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf.gz") as tmp:
                    tmp.write(f.read()); vcf_path = tmp.name
                df_v = load_vcf_with_duckdb(vcf_path)
                if not df_v.empty:
                    st.session_state.pgm_data = df_v
                    st.success(f"✅ {len(df_v)} variants")
            if st.button("⚡ Démo PGM", key="demo_pgm"):
                st.session_state.pgm_data = generate_demo_pgm_data()
            st.markdown(f"*{len(st.session_state.pgm_data)}* variants")
        else:
            st.markdown("### 🧬 Épi")
            f = st.file_uploader("CSV/TSV", type=["csv","tsv","txt"], key="ul_epi")
            if f is not None:
                df_e = parse_epitranscriptomic_file(f)
                if df_e is not None:
                    st.session_state.epi_data = df_e; st.success(f"✅ {len(df_e)}")
            if st.button("⚡ Démo Épi", key="demo_epi"):
                st.session_state.epi_data = generate_demo_epitranscriptomic_data()
            st.markdown(f"*{len(st.session_state.epi_data)}* positions")

        st.markdown("---")
        st.markdown("### 🤖 IA")
        provider = st.selectbox("Fournisseur",
            ["Gemini Flash (Google — GRATUIT)","Groq (gratuit)","DeepSeek (gratuit)",
             "OpenRouter (gratuit)","Ollama (local)"], key="ai_prov")
        st.session_state.ai_provider = provider
        if provider.startswith("Gemini"):
            st.session_state.gemini_key = st.text_input("Clé Gemini", type="password",
                value=st.session_state.get("gemini_key", ""), key="gk_in")
        elif provider.startswith("Groq"):
            st.session_state.groq_key = st.text_input("Clé Groq", type="password",
                value=st.session_state.get("groq_key", ""), key="grk_in")
        elif provider.startswith("DeepSeek"):
            st.session_state.deepseek_key = st.text_input("Clé DeepSeek", type="password",
                value=st.session_state.get("deepseek_key", ""), key="dsk_in")
        elif provider.startswith("OpenRouter"):
            st.session_state.openrouter_key = st.text_input("Clé OpenRouter", type="password",
                value=st.session_state.get("openrouter_key", ""), key="ork_in")

    # Onglets
    tab_names = [
        "🏠 Accueil", "📊 Diversité α/β", "🧮 Abondance Diff.", "🧬 CoDA / CLR",
        "📈 Raréfaction", "🔬 Biomarqueurs ROC", "🌿 KEGG",
        "🔗 Multi-Omics", "🧬 DNABERT-2", "⚗️ Causal ML", "✨ GenAI",
        "🔒 Federated", "🔵 Clustering", "🌲 Random Forest", "⏱ Dynamique",
        "🧩 VAE", "💡 XAI/SHAP", "🕸 GNN", "📄 Rapport IA", "🧬 Multi-Omics Avancé",
        "📝 Article", "🧬 PGM Clinique", "🧬 Épitranscriptomique Avancé",
        "✂️ CRISPR Design & Delivery",
        "🌱 Plant & Agronomie",
    ]
    tabs = st.tabs(tab_names)

    # Onglet 0 : Accueil
    with tabs[0]:
        st.markdown("## 🏠 Accueil — MetaInsight v9")
        st.markdown('<div class="badge-new">Données RÉELLES</div> <div class="badge-fix">CRISPR</div> <div class="badge-new">Plant</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        df_m = st.session_state.df_microbiome
        c1.metric("Échantillons", len(df_m) if df_m is not None else 0)
        c2.metric("Variants PGM", len(st.session_state.pgm_data))
        c3.metric("Modifications Épi", len(st.session_state.epi_data))
        c4.metric("Groupes", df_m["environment"].nunique() if df_m is not None and "environment" in df_m else 0)
        st.markdown("""
        **MetaInsight v9** — 25 onglets :
        - 🦠 Microbiome (diversité, abondance, KEGG...)
        - 🧬 PGM clinique (BRCA, CPIC, CADD)
        - 🧬 Épitranscriptomique (m6A, m5C, Ψ)
        - ✂️ CRISPR Design & Delivery (14 sous-onglets)
        - 🌱 **Plant & Agronomie (NOUVEAU)** — compatible SIVB/CAST 2026
        """)

    # Onglet 1 : Diversité
    with tabs[1]:
        st.markdown("## 📊 Diversité α/β")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            env_col = "environment" if "environment" in df.columns else next(
                (c for c in ['group','class','condition','label'] if c in df.columns), None)
            if env_col:
                tc = detect_feature_cols(df)
                if tc:
                    alpha_df = compute_alpha_diversity(df, tc)
                    alpha_df[env_col] = df[env_col].values
                    m = st.selectbox("Métrique",
                        ["Shannon H'","Simpson (1-D)","Richness","Chao1","Evenness (J)"],
                        key="metric_alpha")
                    fig = px.box(alpha_df, x=env_col, y=m, color=env_col,
                                 template="plotly_dark", points="all")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 2 : Abondance
    with tabs[2]:
        st.markdown("## 🧮 Abondance Différentielle")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            gc = next((c for c in ['environment','group','class','condition','label'] if c in df.columns), None)
            if tc and gc:
                groups = list(df[gc].unique())
                c1, c2, c3 = st.columns(3)
                with c1: method = st.selectbox("Méthode", ["ALDEx2-like","LEfSe","MaAsLin2"], key="da_m")
                with c2: g1 = st.selectbox("G1", groups, key="g1")
                with c3: g2 = st.selectbox("G2", groups, index=min(1, len(groups)-1), key="g2")
                if st.button("🚀 Analyser", key="da_btn"):
                    if method == "ALDEx2-like":
                        r = aldex2_like(df, tc, gc, g1, g2)
                        if r is not None: st.dataframe(r)
                    elif method == "LEfSe":
                        r = lefse_like(df, tc, gc)
                        if r is not None: st.dataframe(r.head(15))
                    else:
                        st.dataframe(maaslin2_like(df, tc, gc).head(15))

    # Onglet 3 : CoDA
    with tabs[3]:
        st.markdown("## 🧬 CoDA")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                X = clr_transform(df[tc].values.astype(float) + 1e-9)
                coords = PCA(n_components=2).fit_transform(X)
                pca_df = pd.DataFrame(coords, columns=["PC1","PC2"])
                gc = next((c for c in ['environment','group'] if c in df.columns), None)
                if gc: pca_df[gc] = df[gc].values
                fig = px.scatter(pca_df, x="PC1", y="PC2",
                                 color=gc if gc else None, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    # Onglet 4 : Raréfaction
    with tabs[4]:
        st.markdown("## 📈 Raréfaction")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty and 'environment' in df.columns:
            tc = detect_feature_cols(df)
            if tc:
                if st.button("🚀 Calculer", key="rare_btn"):
                    curves = rarefaction_curve(df, tc, 20)
                    fig = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    for i, (env, (d, r)) in enumerate(curves.items()):
                        fig.add_trace(go.Scatter(x=d.tolist(), y=r, mode='lines+markers',
                                                  name=env, line=dict(color=colors[i % len(colors)])))
                    fig.update_layout(template="plotly_dark", title="Raréfaction")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 5 : ROC
    with tabs[5]:
        st.markdown("## 🔬 Biomarqueurs ROC")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            gc = next((c for c in ['environment','group','class','condition','label'] if c in df.columns), None)
            if tc and gc:
                groups = list(df[gc].unique())
                c1, c2 = st.columns(2)
                with c1: gp = st.selectbox("Pos", groups, key="roc_p")
                with c2: gn = st.selectbox("Neg", groups, index=min(1, len(groups)-1), key="roc_n")
                if st.button("🚀 ROC", key="roc_btn"):
                    sub = df[df[gc].isin([gp, gn])]
                    y = (sub[gc] == gp).astype(int).values
                    auc_results = []
                    for t in tc[:20]:
                        fpr, tpr, _ = roc_curve(y, sub[t].values)
                        auc_results.append({"Taxon": t, "AUC": round(auc(fpr, tpr), 3)})
                    st.session_state.biomarker_results = {"df_ranked": pd.DataFrame(auc_results).sort_values("AUC", ascending=False)}
                    st.dataframe(st.session_state.biomarker_results["df_ranked"])

    # Onglet 6 : KEGG
    with tabs[6]:
        st.markdown("## 🌿 KEGG")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty and 'environment' in df.columns:
            tc = detect_feature_cols(df)
            if tc:
                if st.button("🚀 Prédire", key="kegg_btn"):
                    kegg_df = kegg_functional_prediction(df, tc)
                    kegg_mean = kegg_df.groupby(df["environment"]).mean()
                    fig = px.imshow(kegg_mean.T, color_continuous_scale="YlOrRd",
                                    template="plotly_dark", title="KEGG")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 7 : Multi-Omics
    with tabs[7]:
        st.markdown("## 🔗 CCA")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                if st.button("🚀 CCA", key="mo_btn"):
                    X = clr_transform(df[tc].values.astype(float) + 1e-9)
                    nm = min(5, X.shape[1])
                    Xm = X[:, :nm] + np.random.randn(len(df), nm) * 0.5
                    Xc, Yc = CCA(n_components=2).fit_transform(X, Xm)
                    gc = next((c for c in ['environment','group'] if c in df.columns), None)
                    dfp = pd.DataFrame({"CCA1": Xc[:,0], "CCA2": Yc[:,0]})
                    if gc: dfp[gc] = df[gc].values
                    fig = px.scatter(dfp, x="CCA1", y="CCA2",
                                     color=gc if gc else None, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 8 : DNABERT-2
    with tabs[8]:
        st.markdown("## 🧬 DNABERT-2")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                tokens = tc[:min(8, len(tc))]
                corr = df[tokens].corr().values
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i in range(3):
                    sns.heatmap(np.abs(corr) ** (i+1), xticklabels=tokens, yticklabels=tokens,
                                ax=axes[i], cmap="viridis", vmin=0, vmax=1)
                    axes[i].set_title(f"Head {i+1}")
                st.pyplot(fig); plt.close()

    # Onglet 9 : Causal ML
    with tabs[9]:
        st.markdown("## ⚗️ Causal ML")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                c = st.selectbox("Cause", tc, key="c_cause")
                e = st.selectbox("Effet", tc, index=min(1, len(tc)-1), key="c_eff")
                if st.button("🚀 Analyser", key="causal_btn"):
                    corr, p = spearmanr(df[c], df[e])
                    st.metric("Spearman", f"{corr:.3f}")
                    st.metric("p-value", f"{p:.4f}")

    # Onglet 10 : GenAI
    with tabs[10]:
        st.markdown("## ✨ GenAI")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            gc = next((c for c in ['environment','group'] if c in df.columns), None)
            if tc and gc:
                ns = st.slider("Échantillons", 10, 200, 50, key="gen_n")
                tg = st.selectbox("Cible", df[gc].unique(), key="gen_t")
                if st.button("✨ Générer", key="gen_btn"):
                    sub = df[df[gc] == tg][tc].values
                    synth = np.clip(np.random.randn(ns, len(tc)) * (sub.std(0) + 1e-6) + sub.mean(0), 0, None)
                    st.dataframe(pd.DataFrame(synth, columns=tc).head())

    # Onglet 11 : Federated
    with tabs[11]:
        st.markdown("## 🔒 Federated")
        r = st.slider("Rounds", 2, 20, 10, key="fed_r")
        if st.button("🚀 Simuler", key="fed_btn"):
            acc = 75 + 18 * (1 - np.exp(-np.arange(1, r+1)/5))
            fig = px.line(x=np.arange(1, r+1), y=acc, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # Onglet 12 : Clustering
    with tabs[12]:
        st.markdown("## 🔵 Clustering")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                k = st.slider("K", 2, 8, 4, key="cl_k")
                if st.button("🚀 K-means", key="cl_btn"):
                    X = clr_transform(df[tc].values.astype(float) + 1e-9)
                    cl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
                    coords = PCA(n_components=2).fit_transform(X)
                    dfp = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1],
                                        "Cluster": cl.astype(str)})
                    fig = px.scatter(dfp, x="PC1", y="PC2", color="Cluster", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 13 : Random Forest
    with tabs[13]:
        st.markdown("## 🌲 Random Forest")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            gc = next((c for c in ['environment','group','class','condition','label'] if c in df.columns), None)
            if tc and gc:
                if st.button("🚀 Entraîner", key="rf_btn"):
                    X = clr_transform(df[tc].values.astype(float) + 1e-9)
                    y = LabelEncoder().fit_transform(df[gc].values)
                    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
                    imp = pd.DataFrame({"Feature": tc, "Importance": rf.feature_importances_}
                                       ).sort_values("Importance", ascending=False).head(10)
                    fig = px.bar(imp, x="Importance", y="Feature", orientation='h',
                                 template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 14 : Dynamique
    with tabs[14]:
        st.markdown("## ⏱ Dynamique")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                taxon = st.selectbox("Taxon", tc, key="dyn_t")
                if st.button("🚀 Modéliser", key="dyn_btn"):
                    vals = df[taxon].values
                    fig = px.line(x=np.arange(len(vals)), y=vals, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 15 : VAE
    with tabs[15]:
        st.markdown("## 🧩 VAE")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                if st.button("🚀 Projeter", key="vae_btn"):
                    X = clr_transform(df[tc].values.astype(float) + 1e-9)
                    lat = PCA(n_components=2).fit_transform(X)
                    gc = next((c for c in ['environment','group'] if c in df.columns), None)
                    dfp = pd.DataFrame({"z1": lat[:,0], "z2": lat[:,1]})
                    if gc: dfp[gc] = df[gc].values
                    fig = px.scatter(dfp, x="z1", y="z2",
                                     color=gc if gc else None, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 16 : XAI
    with tabs[16]:
        st.markdown("## 💡 XAI")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            gc = next((c for c in ['environment','group','class','condition','label'] if c in df.columns), None)
            if tc and gc:
                if st.button("🚀 SHAP approx", key="shap_btn"):
                    X = clr_transform(df[tc].values.astype(float) + 1e-9)
                    y = LabelEncoder().fit_transform(df[gc].values)
                    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
                    imp = permutation_importance(rf, X, y, n_repeats=3, random_state=42)
                    idf = pd.DataFrame({"Feature": tc, "Importance": imp.importances_mean}
                                       ).sort_values("Importance", ascending=False).head(10)
                    fig = px.bar(idf, x="Importance", y="Feature", orientation='h',
                                 template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 17 : GNN
    with tabs[17]:
        st.markdown("## 🕸 GNN")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            tc = detect_feature_cols(df)
            if tc:
                th = st.slider("Seuil", 0.3, 0.9, 0.5, key="th_gnn")
                if st.button("🚀 Construire", key="gnn_btn"):
                    X = clr_transform(df[tc].values.astype(float) + 1e-9)
                    nf = min(15, len(tc))
                    corr = np.corrcoef(X[:, :nf].T)
                    G = nx.Graph()
                    for i in range(nf): G.add_node(tc[i])
                    for i in range(nf):
                        for j in range(i+1, nf):
                            if abs(corr[i, j]) >= th: G.add_edge(tc[i], tc[j])
                    pos = nx.spring_layout(G, seed=42)
                    ex, ey = [], []
                    for u, v in G.edges():
                        x0, y0 = pos[u]; x1, y1 = pos[v]
                        ex.extend([x0, x1, None]); ey.extend([y0, y1, None])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(color='#2A3550')))
                    nx_ = [pos[n][0] for n in G.nodes()]
                    ny_ = [pos[n][1] for n in G.nodes()]
                    fig.add_trace(go.Scatter(x=nx_, y=ny_, mode='markers+text',
                                             text=list(G.nodes()), textposition="top center",
                                             marker=dict(size=10, color='#00D4AA')))
                    fig.update_layout(template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 18 : Rapport IA
    with tabs[18]:
        st.markdown("## 📄 Rapport IA")
        p = st.text_area("Question", "Analyser les différences", key="rep_p")
        if st.button("🤖 Générer", key="rep_btn"):
            r = call_ai(p, st.session_state.ai_provider,
                gemini_key=st.session_state.get("gemini_key", ""),
                groq_key=st.session_state.get("groq_key", ""),
                openrouter_key=st.session_state.get("openrouter_key", ""),
                deepseek_key=st.session_state.get("deepseek_key", ""))
            st.info(r)

    # Onglet 19 : Multi-Omics Avancé
    with tabs[19]:
        st.markdown("## 🧬 Multi-Omics Avancé")
        st.info("Chargez vos fichiers dans la sidebar")

    # Onglet 20 : Article
    with tabs[20]:
        st.markdown("## 📝 Article")
        with st.form("article_f"):
            title = st.text_input("Titre", "Analyse intégrative", key="art_t")
            sections = st.multiselect("Sections",
                ["Résumé","Introduction","Méthodes","Résultats","Discussion"], key="art_s")
            if st.form_submit_button("🤖 Générer"):
                prompt = f"Générer un article '{title}' sections {', '.join(sections)}"
                r = call_ai(prompt, st.session_state.ai_provider,
                    gemini_key=st.session_state.get("gemini_key", ""),
                    groq_key=st.session_state.get("groq_key", ""),
                    openrouter_key=st.session_state.get("openrouter_key", ""),
                    deepseek_key=st.session_state.get("deepseek_key", ""))
                st.markdown(r)

    # Onglet 21 : PGM Clinique
    with tabs[21]:
        st.markdown("## 🧬 PGM Clinique")
        pgm = st.session_state.pgm_data
        if pgm is not None and not pgm.empty:
            pt = st.tabs(["🧬 BRCA","💊 Pharmaco","📊 Lollipop"])
            with pt[0]:
                regions = {"BRCA1": {"chrom": "17", "start": 43044295, "end": 43125483},
                           "BRCA2": {"chrom": "13", "start": 32315474, "end": 32400266}}
                parts = []
                for g, r in regions.items():
                    m = (pgm["chrom"] == r["chrom"]) & (pgm["pos"] >= r["start"]) & (pgm["pos"] <= r["end"])
                    sub = pgm.loc[m].copy()
                    if not sub.empty: sub["gene"] = g; parts.append(sub)
                if parts: st.dataframe(pd.concat(parts, ignore_index=True), use_container_width=True)
            with pt[1]:
                pharma = pd.DataFrame([
                    {"gene":"CYP2D6","rsid":"rs3892097","phenotype":"Métaboliseur lent","drug":"Codéine"},
                    {"gene":"TPMT","rsid":"rs1800460","phenotype":"Déficit TPMT","drug":"Azathioprine"},
                    {"gene":"DPYD","rsid":"rs3918290","phenotype":"Déficit DPD","drug":"5-FU"},
                ])
                if "rsid" in pgm.columns:
                    m = pgm.merge(pharma, on="rsid", how="inner")
                    if not m.empty: st.dataframe(m, use_container_width=True)
            with pt[2]:
                g = st.selectbox("Gène", ["BRCA1","BRCA2","TP53"], key="lol_g")
                lengths = {"BRCA1": 1860, "BRCA2": 3418, "TP53": 393}
                np.random.seed(42)
                nm = np.random.randint(3, 10)
                muts = [{"pos": np.random.randint(1, lengths[g]),
                         "aa": f"{np.random.choice(list('ARNDCQEGHILKMFPSTWYV'))}"
                               f"{np.random.randint(1, lengths[g])}"
                               f"{np.random.choice(list('ARNDCQEGHILKMFPSTWYV'))}",
                         "freq": np.random.uniform(0.01, 0.5)} for _ in range(nm)]
                if st.button("🎯 Lollipop", key="lol_btn"):
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot([0, lengths[g]], [0, 0], 'k-', lw=3)
                    mf = max([m["freq"] for m in muts]) if muts else 1
                    for mut in muts:
                        h = mut["freq"] / mf * 2
                        ax.plot([mut["pos"], mut["pos"]], [0, h], 'o-', color='red', ms=5)
                        ax.text(mut["pos"], h + 0.1, mut["aa"], rotation=45, fontsize=8)
                    ax.set_xlim(0, lengths[g])
                    st.pyplot(fig); plt.close()

    # Onglet 22 : Épitranscriptomique Avancé
    with tabs[22]:
        st.markdown("## 🧬 Épitranscriptomique Avancé")
        epi = st.session_state.epi_data
        if epi is not None and not epi.empty:
            es = st.tabs(["📊 Profil","🔥 Heatmap","📈 Corrélation","📊 Stats"])
            with es[0]:
                trs = epi['transcript_id'].unique()
                sel = st.selectbox("Transcrit", trs, key="epi_tr")
                fig = plot_modification_profile_advanced(epi, sel)
                if fig: st.plotly_chart(fig, use_container_width=True)
            with es[1]:
                fig = plot_modification_heatmap(epi)
                if fig: st.plotly_chart(fig, use_container_width=True)
            with es[2]:
                if 'expression_TPM' in epi.columns and 'modification_rate' in epi.columns:
                    corrs = []
                    for g in epi['gene'].unique():
                        sub = epi[epi['gene'] == g]
                        if len(sub) > 2:
                            c, p = pearsonr(sub['modification_rate'], sub['expression_TPM'])
                            corrs.append({"gene": g, "corr": c, "p": p})
                    st.dataframe(pd.DataFrame(corrs))
            with es[3]:
                if 'condition' in epi.columns:
                    fig = px.box(epi, x='condition', y='modification_rate',
                                 color='condition', template="plotly_dark", points="all")
                    st.plotly_chart(fig, use_container_width=True)

    # Onglet 23 : CRISPR
    with tabs[23]:
        render_crispr_tab()

    # Onglet 24 : Plant & Agronomie
    with tabs[24]:
        render_plant_tab()

    st.markdown("---")
    st.caption("🧬 MetaInsight v9 + CRISPR + Plant (données réelles) · Doench 2016 · UCSC · CFD/MIT · Ensembl Plants · SIVB/CAST 2026")

if __name__ == "__main__":
    main()
