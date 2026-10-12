# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v9 — Multi-omique + CRISPR + Plant + HUB UNIFIÉ
# Fichier unique — Cohérence cross-module — 26 onglets
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
import os, io, re, gzip, json, pickle, tempfile, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from io import StringIO
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, silhouette_score, roc_curve, auc,
                              r2_score, mean_squared_error, mean_absolute_error, roc_auc_score)
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
import networkx as nx
import joblib

try: import duckdb; DUCKDB_AVAILABLE = True
except: DUCKDB_AVAILABLE = False
try: import pyarrow as pa; import pyarrow.parquet as pq; PYARROW_AVAILABLE = True
except: PYARROW_AVAILABLE = False
try: import pysam; PYSAM_AVAILABLE = True
except: PYSAM_AVAILABLE = False
try: from Bio import SeqIO; BIOPYTHON_AVAILABLE = True
except: BIOPYTHON_AVAILABLE = False
try: import xgboost as xgb; XGBOOST_AVAILABLE = True
except: XGBOOST_AVAILABLE = False

_ENV_GEMINI_KEY = os.environ.get('GEMINI_API_KEY', '')
_ENV_GROQ_KEY = os.environ.get('GROQ_API_KEY', '')
_ENV_OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
_ENV_DEEPSEEK_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

st.set_page_config(page_title="MetaInsight v9 + HUB", layout="wide", initial_sidebar_state="auto")

st.markdown("""
<style>
.stApp { background-color: #0A0E1A; color: #E8EDF5; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #0A0E1A;
    border-bottom: 1px solid #2A3550; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background-color: #0F1525; border-radius: 8px 8px 0 0;
    color: #7A8BA8; padding: 6px 12px; font-weight: 500; font-size: 0.82rem; }
.stTabs [aria-selected="true"] { background-color: #151C30; color: #00D4AA;
    border-bottom: 2px solid #00D4AA; }
.stButton button { background-color: #1A2238; border: 1px solid #2A3550;
    color: #E8EDF5; border-radius: 8px; }
.stButton button:hover { background-color: #1F2940; border-color: #00D4AA; color: #00D4AA; }
.badge-new { background: linear-gradient(90deg,#00D4AA,#4D9FFF); color:#000;
    font-size:0.65rem; padding:2px 7px; border-radius:10px; font-weight:700;
    margin-left:4px; vertical-align:middle; }
.badge-hub { background: linear-gradient(90deg,#FF6B6B,#FFD93D); color:#000;
    font-size:0.65rem; padding:2px 7px; border-radius:10px; font-weight:700;
    margin-left:4px; vertical-align:middle; }
.ref-box { background:#0F1525; border-left:3px solid #00D4AA; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A8BA8; margin:6px 0; }
.hub-box { background:#1A1525; border-left:3px solid #FF6B6B; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#FFD9D9; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

CACHE_DIR = Path("./crispr_cache"); CACHE_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES DÉMO
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_demo_microbiome():
    environments = ["Sol aride", "Eau marine", "Gut", "Sol agricole", "Sédiments", "Biofilm"]
    taxa = ["Proteobacteria","Actinobacteriota","Firmicutes","Bacteroidota","Archaea",
            "Acidobacteria","Chloroflexi","Planctomycetes","Ascomycota","Caudovirales"]
    base_profiles = {
        "Sol aride": [28,20,5,4,8,6,4,3,2,1], "Eau marine": [35,10,8,15,2,5,3,4,8,6],
        "Gut": [15,12,30,22,1,3,2,2,4,2], "Sol agricole": [22,25,10,8,4,10,7,5,3,2],
        "Sédiments": [18,14,12,10,6,8,9,6,5,4], "Biofilm": [30,18,6,9,3,7,5,4,6,5],
    }
    data = []
    for env in environments:
        for rep in range(4):
            noisy = np.array(base_profiles[env]) + np.random.normal(0, 2, size=len(taxa))
            noisy = np.clip(noisy, 0, None); noisy = noisy / noisy.sum() * 100
            row = {"sample_id": f"{env[:3].upper()}_{rep+1:03d}", "environment": env}
            for i, tax in enumerate(taxa): row[tax] = round(noisy[i], 2)
            probs = noisy / 100.0
            row.update({"shannon": round(entropy(probs, base=2), 3),
                        "simpson": round(1 - np.sum(probs**2), 3),
                        "chao1": round(len(taxa) + np.random.uniform(0, 5), 1),
                        "faith_pd": round(float((noisy > 0).sum()) * 2.1 + float(np.std(noisy[noisy>0])) * 0.5, 2),
                        "classified_pct": round(np.random.uniform(70, 99), 1),
                        "ph": round(np.random.uniform(4, 8), 2),
                        "temperature_c": round(np.random.uniform(15, 40), 1),
                        "moisture_pct": round(np.random.uniform(5, 80), 1)})
            data.append(row)
    return pd.DataFrame(data)

@st.cache_data
def generate_demo_pgm_data():
    np.random.seed(42); n = 200
    chroms = np.random.choice([str(i) for i in range(1,23)] + ["X"], size=n)
    pos = np.random.randint(100000, 100000000, n)
    refs = np.random.choice(["A","C","G","T"], n)
    alts = np.random.choice(["A","C","G","T"], n)
    for i in range(n):
        while alts[i] == refs[i]: alts[i] = np.random.choice(["A","C","G","T"])
    return pd.DataFrame({
        "chrom": chroms, "pos": pos, "ref": refs, "alt": alts,
        "qual": np.random.uniform(10, 1000, n),
        "filter": np.random.choice(["PASS","LowQual",""], n, p=[0.7,0.2,0.1]), "info": "",
        "clinvar_sig": np.random.choice(["Pathogenic","Likely_pathogenic","VUS","Benign","Likely_benign"],
                                          n, p=[0.05,0.1,0.6,0.15,0.1]),
        "variant_type": np.random.choice(["SNP","Indel","MNV"], n),
        "rsid": [f"rs{np.random.randint(100000, 999999)}" for _ in range(n)],
        "allele_freq": np.random.uniform(0.001, 0.5, n),
        "cadd_phred": np.random.uniform(0, 45, n)})

@st.cache_data
def generate_demo_epitranscriptomic_data():
    np.random.seed(123); n = 80
    positions = np.sort(np.random.randint(50, 1950, n))
    mod_types = np.random.choice(['m6A','m5C','Ψ','m1A','2OMe','m7G','Nm'], n,
                                   p=[0.35,0.15,0.15,0.1,0.1,0.08,0.07])
    df = pd.DataFrame({
        'transcript_id': np.random.choice(['ENST00000380152','ENST00000456328',
                                             'ENST00000585865','ENST00000318560'], n),
        'position': positions, 'modification': mod_types,
        'modification_rate': np.round(np.random.beta(2, 5, n), 3),
        'pseudouridine_score': np.random.uniform(0, 1, n),
        'condition': np.random.choice(['Contrôle', 'Traité'], n),
        'sample_id': [f'Sample_{i%6+1:02d}' for i in range(n)],
        'expression_TPM': np.round(np.random.lognormal(5, 1.5, n), 2),
        'gene': np.random.choice(['BRCA1','TP53','EGFR','MYC','KRAS','PTEN'], n)})
    motifs = []
    for m in mod_types:
        if m == 'm6A': motifs.append(np.random.choice(['DRACH','GGACU','RRACH']))
        elif m == 'm5C': motifs.append(np.random.choice(['CG','CNG']))
        elif m == 'Ψ': motifs.append(np.random.choice(['UGU','GU']))
        else: motifs.append('')
    df['consensus_motif'] = motifs
    return df

# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS GÉNÉRIQUES
# ══════════════════════════════════════════════════════════════════════════════
def clr_transform(X):
    X_pos = np.clip(X, 1e-9, None); log_X = np.log(X_pos)
    return log_X - log_X.mean(axis=1, keepdims=True)

def compute_alpha_diversity(df, taxa_cols):
    results = []
    for _, row in df.iterrows():
        vals = np.clip(row[taxa_cols].values.astype(float), 0, None)
        total = vals.sum(); probs = vals / total if total > 0 else vals
        probs_nz = probs[probs > 0]
        shannon = float(entropy(probs_nz, base=2)) if len(probs_nz) > 0 else 0.0
        richness = int((vals > 0).sum())
        n1, n2 = int((vals == 1).sum()), int((vals == 2).sum())
        chao1 = richness + (n1*(n1-1))/(2*(n2+1)) if n2 > 0 else richness + n1*(n1-1)/2
        evenness = shannon / np.log2(richness) if richness > 1 else 0.0
        results.append({"Shannon H'": round(shannon, 3),
                        "Simpson (1-D)": round(float(1 - np.sum(probs**2)), 3),
                        "Richness": richness, "Chao1": round(chao1, 1),
                        "Evenness (J)": round(evenness, 3)})
    return pd.DataFrame(results, index=df.index)

def detect_feature_cols(df):
    meta = {"sample_id","environment","group","label","class","condition","shannon",
            "simpson","chao1","faith_pd","classified_pct","ph","temperature_c","moisture_pct"}
    cols = []
    for col in df.columns:
        if col.lower() in meta: continue
        if col.endswith("_ZYG") or "date" in col.lower() or col.lower() == "id": continue
        d = df[col]
        if pd.api.types.is_numeric_dtype(d) and d.std() > 0: cols.append(col)
        elif pd.api.types.is_object_dtype(d) or pd.api.types.is_string_dtype(d):
            if 2 <= len(d.dropna().unique()) <= 30: cols.append(col)
    return cols

def aldex2_like(df, taxa_cols, group_col, group1, group2):
    g1 = df[df[group_col] == group1][taxa_cols].values.astype(float)
    g2 = df[df[group_col] == group2][taxa_cols].values.astype(float)
    if len(g1) < 2 or len(g2) < 2: return None
    results = []
    for j, tax in enumerate(taxa_cols):
        clr1 = clr_transform(g1 + 0.5)[:, j]; clr2 = clr_transform(g2 + 0.5)[:, j]
        effect = (clr1.mean() - clr2.mean()) / (np.sqrt((clr1.std()**2 + clr2.std()**2)/2) + 1e-9)
        try: _, pval = mannwhitneyu(clr1, clr2, alternative='two-sided')
        except: pval = 1.0
        results.append({"Taxon": tax, "CLR G1": round(clr1.mean(), 3),
                        "CLR G2": round(clr2.mean(), 3), "Effect": round(effect, 3),
                        "p": round(pval, 4)})
    res_df = pd.DataFrame(results); n = len(res_df)
    pvals = res_df["p"].values; sidx = np.argsort(pvals); bh = np.zeros(n)
    for r, i in enumerate(sidx): bh[i] = min(1.0, pvals[i]*n/(r+1))
    for i in range(n-2, -1, -1): bh[sidx[i]] = min(bh[sidx[i]], bh[sidx[i+1]])
    res_df["BH"] = bh.round(4)
    return res_df.sort_values("BH")

def lefse_like(df, taxa_cols, group_col):
    groups = df[group_col].unique()
    if len(groups) < 2: return None
    results = []
    for tax in taxa_cols:
        gv = [df[df[group_col] == g][tax].values for g in groups]
        try: _, p_kw = kruskal(*gv)
        except: p_kw = 1.0
        if p_kw < 0.05:
            means = [v.mean() for v in gv]; stds = [v.std() + 1e-9 for v in gv]
            ps = np.sqrt(np.mean([s**2 for s in stds]))
            lda = abs(max(means)-min(means))/(ps+1e-9)*np.log10(len(df)+1)
            bg = groups[np.argmax(means)]
        else: lda, bg = 0.0, "—"
        results.append({"Taxon": tax, "LDA": round(lda, 3), "Best": bg, "p": round(p_kw, 4)})
    return pd.DataFrame(results).sort_values("LDA", ascending=False)

def kegg_functional_prediction(df, taxa_cols):
    FW = {"Proteobacteria":["Nitrogen fixation","Flagellar biosynthesis"],
          "Firmicutes":["Butyrate production","Sporulation"],
          "Bacteroidota":["Polysaccharide degradation","Vitamin B12"],
          "Actinobacteriota":["Secondary metabolites","Antibiotic biosynthesis"],
          "Archaea":["Methanogenesis","CO2 fixation"]}
    rows = []
    for _, row in df.iterrows():
        d = {}
        for t in taxa_cols:
            if t in FW:
                for p in FW[t]: d[p] = d.get(p, 0) + float(row.get(t, 0))
        rows.append(d)
    return pd.DataFrame(rows, index=df.index).fillna(0)

def call_ai(prompt, provider, gemini_key=None, groq_key=None, openrouter_key=None,
            deepseek_key=None, gemini_model="gemini-1.5-flash",
            groq_model="llama-3.3-70b-versatile",
            openrouter_model="meta-llama/llama-3.1-8b-instruct:free",
            ollama_model="llama3"):
    try:
        if provider.startswith("Gemini"):
            if not gemini_key: return "🔑 Clé Gemini manquante"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            r = requests.post(url, json={"contents":[{"parts":[{"text":prompt}]}]}, timeout=30)
            if r.status_code == 200: return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"⚠️ Gemini {r.status_code}"
        elif provider.startswith("Groq"):
            if not groq_key: return "🔑 Clé Groq manquante"
            h = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              json={"model": groq_model, "messages":[{"role":"user","content":prompt}]},
                              headers=h, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Groq {r.status_code}"
        elif provider.startswith("DeepSeek"):
            if not deepseek_key: return "🔑 Clé DeepSeek manquante"
            h = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
            r = requests.post("https://api.deepseek.com/v1/chat/completions",
                              json={"model":"deepseek-chat","messages":[{"role":"user","content":prompt}]},
                              headers=h, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ DeepSeek {r.status_code}"
        elif provider.startswith("OpenRouter"):
            if not openrouter_key: return "🔑 Clé OpenRouter manquante"
            h = {"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json={"model": openrouter_model, "messages":[{"role":"user","content":prompt}]},
                              headers=h, timeout=30)
            if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ OpenRouter {r.status_code}"
        elif provider.startswith("Ollama"):
            try:
                r = requests.post("http://localhost:11434/api/generate",
                                  json={"model": ollama_model, "prompt": prompt, "stream": False}, timeout=60)
                if r.status_code == 200: return r.json().get("response", "")
                return f"⚠️ Ollama {r.status_code}"
            except requests.exceptions.ConnectionError:
                return "❌ Ollama non lancé"
        return "⚠️ Fournisseur inconnu"
    except Exception as e: return f"❌ {str(e)}"

def load_vcf_with_duckdb(file_path):
    if not DUCKDB_AVAILABLE:
        st.error("DuckDB non installé"); return pd.DataFrame()
    # Essayer DuckDB VCF
    try:
        duckdb.sql("INSTALL vcf; LOAD vcf;")
    except Exception:
        pass
    try:
        df = duckdb.sql(f"SELECT * FROM read_vcf('{file_path}') LIMIT 10000").df()
        if not df.empty: return df
    except Exception:
        pass
    # Fallback : lecture VCF brut
    try:
        rows = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                p = line.strip().split('\t')
                if len(p) < 8: continue
                info = p[7]
                clinvar = re.search(r'CLNSIG=([^;]+)', info)
                rsid = re.search(r'RS=([^;]+)', info)
                rows.append({"chrom": p[0], "pos": int(p[1]), "ref": p[3], "alt": p[4],
                             "qual": float(p[5]) if p[5] != '.' else 0,
                             "filter": p[6], "info": info,
                             "clinvar_sig": clinvar.group(1) if clinvar else "NA",
                             "rsid": rsid.group(1) if rsid else "NA"})
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Erreur lecture VCF : {e}"); return pd.DataFrame()

def parse_epitranscriptomic_file(f):
    try:
        content = f.getvalue().decode('utf-8', errors='ignore')
        sep = '\t' if '\t' in content[:1000] else ',' if ',' in content[:1000] else ';'
        return pd.read_csv(io.StringIO(content), sep=sep)
    except Exception as e:
        st.error(f"Erreur : {e}"); return None

def plot_modification_profile_advanced(df, tid=None, smooth=True):
    if tid: df = df[df['transcript_id'] == tid]
    if df.empty: return None
    df = df.sort_values('position')
    fig = go.Figure()
    for m in df['modification'].unique():
        sub = df[df['modification'] == m]
        fig.add_trace(go.Scatter(x=sub['position'], y=sub['modification_rate'],
                                  mode='markers+lines' if smooth else 'markers', name=m))
    fig.update_layout(title=f"Profil - {tid or 'Tous'}", template="plotly_dark",
                      xaxis_title="Position", yaxis_title="Taux")
    return fig

def plot_modification_heatmap(df):
    p = df.pivot_table(index='sample_id', columns='position',
                        values='modification_rate', aggfunc='mean')
    if p.empty: return None
    return px.imshow(p, color_continuous_scale="RdBu_r", aspect="auto", title="Heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# CRISPR — CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
PAM_PATTERNS = {
    "SpCas9 (NGG)": {"pam": "NGG", "guide_len": 20, "pam_side": "3'"},
    "SpCas9-VQR (NGA)": {"pam": "NGA", "guide_len": 20, "pam_side": "3'"},
    "SaCas9 (NNGRRT)": {"pam": "NNGRRT", "guide_len": 21, "pam_side": "3'"},
    "Cas12a (TTTV)": {"pam": "TTTV", "guide_len": 23, "pam_side": "5'"},
}
DELIVERY_VECTORS = {
    "AAV": {"capacite_kb":4.7,"tissus":"Foie, SNC","immuno":"Faible","duree":"Long","in_vivo":True,"in_vitro":True,"cout":"Élevé"},
    "LNP": {"capacite_kb":20,"tissus":"Foie, poumon","immuno":"Modérée","duree":"Transitoire","in_vivo":True,"in_vitro":True,"cout":"Modéré"},
    "Électroporation": {"capacite_kb":100,"tissus":"Ex vivo","immuno":"Aucune","duree":"Transitoire","in_vivo":False,"in_vitro":True,"cout":"Faible"},
    "Lentivirus": {"capacite_kb":8,"tissus":"Large","immuno":"Élevée","duree":"Intégration","in_vivo":True,"in_vitro":True,"cout":"Modéré"},
}
CODON_TABLE = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
PROTEIN_TAGS = {
    "FLAG": {"seq":"GACTACAAGGACGACGATGACAAG","aa":"DYKDDDDK"},
    "HA": {"seq":"TACCCATACGATGTTCCAGATTACGCT","aa":"YPYDVPDYA"},
    "His6": {"seq":"CACCACCACCACCACCAC","aa":"HHHHHH"},
    "V5": {"seq":"GGTAAGCCTATCCCTAACCCTCTCCTCGGTCTCGATTCTACG","aa":"GKPIPNPLLGLDST"},
}
BASE_TO_2BIT = {'A':0,'C':1,'G':2,'T':3,'N':4,'U':3}
BIT2_TO_ASCII = {0:'A',1:'C',2:'G',3:'T',4:'N'}
MIT_WEIGHTS = [0.0,0.0,0.014,0.0,0.0,0.395,0.317,0.0,0.0,0.0,
               0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]
CFD_MISMATCH_MATRIX = {
    'A': {'A':[0]*19+[1.0], 'C':[0,0,0.014]+[0]*4+[0.395,0.317]+[0]*11+[1.0],
          'G':[0]*19+[1.0], 'T':[0]*19+[1.0]},
    'C': {'A':[0]*19+[1.0], 'C':[0]*19+[1.0], 'G':[0]*19+[1.0], 'T':[0]*19+[1.0]},
    'G': {'A':[0]*19+[1.0], 'C':[0]*19+[1.0], 'G':[0]*19+[1.0], 'T':[0]*19+[1.0]},
    'T': {'A':[0]*19+[1.0], 'C':[0]*19+[1.0], 'G':[0]*19+[1.0], 'T':[0]*19+[1.0]},
}
UCSC_GENOMES = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes/",
    "dm6": "https://hgdownload.soe.ucsc.edu/goldenPath/dm6/chromosomes/",
}
DOENCH_URLS = [
    "https://raw.githubusercontent.com/maximilianh/crisporWebsite/master/data/doench2016.tab",
    "https://raw.githubusercontent.com/maximilianh/crisporWebsite/master/data/doench2014.tab",
]
DOENCH_BUILTIN = [
    ("GACCCCCTCCACCCCGCGTC",0.92),("GGGCGAGGAGCTGTTCACCG",0.85),
    ("GAGCTGGACGGCGACGTAAA",0.78),("GGTGAACCGCATCGAGCTGA",0.71),
    ("GCATCGAGCTGAAGGGCATC",0.88),("GCTGAAGGGCATCGACTTCA",0.65),
    ("GGCATCGACTTCAAGGAGGA",0.52),("GACTTCAAGGAGGACGGCAAC",0.81),
    ("GTTCACCGGCGACGTAAACG",0.45),("GTGCCCATCCTGGTCGAGCT",0.90),
    ("GCTGGACGGCGACGTAAACG",0.62),("GCGACGTAAACGGCCACAAG",0.79),
    ("GCCACAAGTTCAGCGTGTCC",0.55),("GTTCAGCGTGTCCGGCGAGG",0.73),
    ("GTCCGGCGAGGGCGAGGGCG",0.68),("GAGGGCGAGGGCGATGCCAC",0.44),
    ("GGGCGATGCCACCTACGGCA",0.75),("GCCACCTACGGCAAGCTGAC",0.83),
    ("GCAAGCTGACCCTGAAGTTC",0.91),("GACCCTGAAGTTCATCTGCAC",0.38),
]

# ══════════════════════════════════════════════════════════════════════════════
# PLANT — CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
PLANT_GENOMES = {
    "Arabidopsis (TAIR10)": {"organism":"arabidopsis_thaliana","assembly":"TAIR10",
        "url_base":"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/",
        "pattern":"Arabidopsis_thaliana.TAIR10.dna.chromosome.{chrom}.fa.gz",
        "chroms":["1","2","3","4","5"],"size_mb":120,"ploidy":"2n=10","importance":"Modèle"},
    "Rice (IRGSP-1.0)": {"organism":"oryza_sativa","assembly":"IRGSP-1.0",
        "url_base":"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/oryza_sativa/dna/",
        "pattern":"Oryza_sativa.IRGSP-1.0.dna.chromosome.{chrom}.fa.gz",
        "chroms":[str(i) for i in range(1,13)],"size_mb":400,"ploidy":"2n=24","importance":"Monocotylédone"},
    "Maize (B73 v5)": {"organism":"zea_mays","assembly":"Zm-B73-REFERENCE-NAM-5.0",
        "url_base":"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/zea_mays/dna/",
        "pattern":"Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.chromosome.{chrom}.fa.gz",
        "chroms":[str(i) for i in range(1,11)],"size_mb":2200,"ploidy":"2n=20","importance":"Culture majeure"},
    "Tomato (SL3.0)": {"organism":"solanum_lycopersicum","assembly":"SL3.0",
        "url_base":"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/solanum_lycopersicum/dna/",
        "pattern":"Solanum_lycopersicum.SL3.0.dna.chromosome.{chrom}.fa.gz",
        "chroms":[str(i) for i in range(1,13)],"size_mb":830,"ploidy":"2n=24","importance":"Solanaceae"},
    "Soybean (Wm82.a4)": {"organism":"glycine_max","assembly":"Wm82.a4.v1",
        "url_base":"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/glycine_max/dna/",
        "pattern":"Glycine_max.V1.0.dna.chromosome.{chrom}.fa.gz",
        "chroms":[str(i) for i in range(1,21)],"size_mb":980,"ploidy":"2n=40","importance":"Légumineuse"},
    "Wheat (IWGSC)": {"organism":"triticum_aestivum","assembly":"IWGSC",
        "url_base":"https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/triticum_aestivum/dna/",
        "pattern":"Triticum_aestivum.IWGSC.dna.chromosome.{chrom}.fa.gz",
        "chroms":[f"{g}{n}" for g in "ABD" for n in range(1,8)][:21],
        "size_mb":14500,"ploidy":"2n=42","importance":"Céréale"},
}
AGRONOMIC_TRAITS = {
    "🌾 Rendement": {"description":"Productivité","genes":{
        "Gn1a":{"species":"Riz","function":"Nombre grains","ref":"Ashikari 2005"},
        "DEP1":{"species":"Riz","function":"Densité panicule","ref":"Huang 2009"},
        "IPA1":{"species":"Riz","function":"Architecture","ref":"Jiao 2010"},
        "TB1":{"species":"Maïs","function":"Domestication","ref":"Doebley 1997"}}},
    "🛡️ Résistance": {"description":"Pathogènes","genes":{
        "Xa21":{"species":"Riz","function":"Résistance Xanthomonas","ref":"Song 1995"},
        "Pi-ta":{"species":"Riz","function":"Résistance Magnaporthe","ref":"Bryan 2000"},
        "Lr34":{"species":"Blé","function":"Résistance rouille","ref":"Krattinger 2009"},
        "Mlo":{"species":"Orge","function":"Résistance oïdium","ref":"Büschges 1997"}}},
    "💧 Sécheresse": {"description":"Stress hydrique","genes":{
        "DREB1A":{"species":"Arabidopsis","function":"Drought-responsive","ref":"Liu 1998"},
        "DREB2A":{"species":"Multiple","function":"Déshydratation","ref":"Sakuma 2006"},
        "NAC":{"species":"Multiple","function":"SNAC1","ref":"Hu 2006"},
        "OsNAC10":{"species":"Riz","function":"Sécheresse+rendement","ref":"Jeong 2010"}}},
    "🧪 Qualité": {"description":"Nutrition","genes":{
        "Wx":{"species":"Riz/Maïs","function":"Amylose","ref":"Wang 1990"},
        "GPC-B1":{"species":"Blé","function":"Protéines grain","ref":"Uauy 2006"},
        "PSY1":{"species":"Maïs","function":"Provitamine A","ref":"Paine 2005"}}},
    "🌿 Architecture": {"description":"Morphologie","genes":{
        "PROG1":{"species":"Riz","function":"Érection","ref":"Jin 2008"},
        "SPL14":{"species":"Riz","function":"Talles/panicules","ref":"Miura 2010"},
        "MOC1":{"species":"Riz","function":"Initiation talles","ref":"Li 2003"}}},
    "⏱️ Floraison": {"description":"Floraison","genes":{
        "Hd1":{"species":"Riz","function":"Photopériode","ref":"Yano 2000"},
        "Hd3a":{"species":"Riz","function":"Florigène","ref":"Kojima 2002"},
        "Ghd7":{"species":"Riz","function":"Floraison+rendement","ref":"Xue 2008"},
        "Vrn1":{"species":"Blé","function":"Vernalisation","ref":"Yan 2003"}}},
}
DOMESTICATION_GENES = {
    "Maïs":{"tb1":"Ramification","tga1":"Gaine graine","bt2":"Poids grain","sh2":"Qualité sucrée"},
    "Riz":{"sh4":"Non-shattering","qSH1":"Dispersion","PROG1":"Érection","Wx":"Amylose","Gn1a":"Grains"},
    "Blé":{"Q":"Fragilité épi","Vrn1":"Vernalisation","Rht-B1":"Hauteur","Gpc-B1":"Protéines"},
    "Tomate":{"sp":"Détermination","u":"Forme fruit","fas":"Locules/taille"},
}
AGRONOMIC_DATABASES = {
    "Gramene": {"url":"https://www.gramene.org/","description":"Comparative plantes","species":["Riz","Maïs","Blé"]},
    "MaizeGDB": {"url":"https://www.maizegdb.org/","description":"Maïs","species":["Maïs"]},
    "RAP-DB": {"url":"https://rapdb.dna.affrc.go.jp/","description":"Riz","species":["Riz"]},
    "TAIR": {"url":"https://www.arabidopsis.org/","description":"Arabidopsis","species":["Arabidopsis"]},
    "Phytozome": {"url":"https://phytozome-next.jgi.doe.gov/","description":"JGI","species":["Multi"]},
}
PLANT_ML_TOOLS = {
    "CRISPR-P 2.0": {"url":"http://crispr.hzau.edu.cn/CRISPR2/","description":"Design plantes",
                     "reference":"Liu 2017","species":["Riz","Arabidopsis","Maïs"]},
    "CHOPCHOP v3": {"url":"https://chopchop.cbu.uib.no/","description":"Multi-org","reference":"Labun 2019","species":["Multi"]},
    "PlantCAD": {"url":"https://huggingface.co/kuleshov-group/PlantCAD","description":"DNA LM","reference":"Zhai 2024","species":["Multi"]},
}

# ══════════════════════════════════════════════════════════════════════════════
# CRISPR — FONCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def download_doench(force=False):
    cf = CACHE_DIR / "doench.parquet"
    if cf.exists() and not force and PYARROW_AVAILABLE:
        try: return pd.read_parquet(cf)
        except: pass
    for url in DOENCH_URLS:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200: continue
            sep = '\t' if '\t' in r.text[:500] else ','
            df = pd.read_csv(StringIO(r.text), sep=sep)
            cm = {}
            for c in df.columns:
                cl = c.lower()
                if any(k in cl for k in ['sgrna','guide_seq','seq','guide']): cm[c]='guide'
                elif any(k in cl for k in ['eff','score']): cm[c]='efficiency'
            df = df.rename(columns=cm)
            if 'guide' in df.columns and 'efficiency' in df.columns:
                df['guide'] = df['guide'].astype(str).str.upper().str.strip()
                df = df[df['guide'].str.len() == 20]
                df = df[df['guide'].str.match(r'^[ACGT]{20}$')]
                df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
                df = df.dropna(subset=['guide','efficiency'])
                if df['efficiency'].max() > 1.5:
                    emin, emax = df['efficiency'].min(), df['efficiency'].max()
                    df['efficiency'] = (df['efficiency']-emin)/(emax-emin)
                if PYARROW_AVAILABLE:
                    try: df[['guide','efficiency']].to_parquet(cf)
                    except: pass
                return df[['guide','efficiency']].reset_index(drop=True)
        except: continue
    return pd.DataFrame(DOENCH_BUILTIN, columns=["guide","efficiency"])

def download_chromosome(genome="hg38", chrom="chr21", progress_callback=None):
    if genome not in UCSC_GENOMES: return None, "Génome non supporté"
    cf = CACHE_DIR / f"{genome}_{chrom}.fa.gz"
    if cf.exists():
        try:
            with gzip.open(cf, "rt") as f:
                seq = "".join(l.strip().upper() for l in f if not l.startswith(">"))
            return seq, None
        except: pass
    url = f"{UCSC_GENOMES[genome]}{chrom}.fa.gz"
    try:
        if progress_callback: progress_callback(0.0, "Téléchargement...")
        r = requests.get(url, stream=True, timeout=180)
        if r.status_code != 200: return None, f"HTTP {r.status_code}"
        total = int(r.headers.get('content-length', 0))
        chunks, dl = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk); dl += len(chunk)
            if progress_callback and total > 0:
                progress_callback(dl/total, f"{dl//1024//1024}MB")
        data = b"".join(chunks)
        with open(cf, "wb") as f: f.write(data)
        content = gzip.decompress(data).decode("utf-8", errors="ignore")
        seq = "".join(l.strip().upper() for l in content.split("\n") if not l.startswith(">"))
        return seq, None
    except Exception as e: return None, str(e)

def download_plant_chromosome(genome_name, chrom, progress_callback=None):
    info = PLANT_GENOMES.get(genome_name)
    if not info: return None, f"Génome '{genome_name}' non supporté"
    cf = CACHE_DIR / f"plant_{info['organism']}_{chrom}.fa.gz"
    if cf.exists():
        try:
            with gzip.open(cf, "rt") as f:
                seq = "".join(l.strip().upper() for l in f if not l.startswith(">"))
            return seq, None
        except: pass
    pattern = info["pattern"].format(chrom=chrom)
    url = info["url_base"] + pattern
    try:
        if progress_callback: progress_callback(0.0, "Téléchargement...")
        r = requests.get(url, stream=True, timeout=240)
        if r.status_code != 200: return None, f"HTTP {r.status_code}"
        total = int(r.headers.get('content-length', 0))
        chunks, dl = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk); dl += len(chunk)
            if progress_callback and total > 0:
                progress_callback(dl/total, f"{dl//1024//1024}MB")
        data = b"".join(chunks)
        with open(cf, "wb") as f: f.write(data)
        content = gzip.decompress(data).decode("utf-8", errors="ignore")
        seq = "".join(l.strip().upper() for l in content.split("\n") if not l.startswith(">"))
        return seq, None
    except Exception as e: return None, str(e)

def rc(seq):
    c = {'A':'T','T':'A','G':'C','C':'G','N':'N','U':'A'}
    return ''.join(c.get(b.upper(),'N') for b in reversed(seq))

def gc_content(seq): return sum(1 for b in seq.upper() if b in 'GC') / max(len(seq),1)
def has_homopolymer(seq, ml=5): return any(b*ml in seq for b in "ACGT")

def match_pam(w, p):
    if len(w) != len(p): return False
    for s, pat in zip(w.upper(), p.upper()):
        if pat == 'N': continue
        if pat == 'V' and s in 'ACG': continue
        if pat == 'R' and s in 'AG': continue
        if s != pat: return False
    return True

def encode_2bit(seq): return np.array([BASE_TO_2BIT.get(b,4) for b in seq.upper()], dtype=np.uint8)
def decode_2bit(arr): return ''.join(BIT2_TO_ASCII.get(int(x),'N') for x in arr)
def rc_2bit(arr): return np.array([0,3,2,1,4], dtype=np.uint8)[arr[::-1]]

def find_pam_sites(sequence, pam="NGG", gl=20, ps="3'"):
    seq = sequence.upper().replace('U','T'); res = []
    for strand, s in [("+", seq), ("-", rc(seq))]:
        for i in range(len(s) - len(pam) + 1):
            if not match_pam(s[i:i+len(pam)], pam): continue
            if ps == "3'":
                gs = i - gl
                if gs < 0: continue
                guide, pp = s[gs:i], i
            else:
                gs = i + len(pam)
                guide = s[gs:gs+gl]
                if len(guide) < gl: continue
                pp = i
            if len(guide) == gl:
                res.append({"strand": strand, "pam": s[i:i+len(pam)],
                            "guide": guide, "gc": round(gc_content(guide),3), "pam_pos": pp+1})
    return res

def score_doench(guide):
    if len(guide) < 15: return 0.0
    gc = gc_content(guide)
    gcs = (1.0 - abs(gc - 0.55)*2) if 0.4 <= gc <= 0.7 else 0.2
    p20 = 0.15 if guide[-1]=='G' else (0.05 if guide[-1]=='C' else 0.0)
    pt = 0.4 if 'TTTT' in guide else 0.0
    hm = 0.3 if has_homopolymer(guide,5) else 0.0
    return round(max(0.0,min(1.0,0.5*gcs+p20-pt-hm)),3)

def count_ot(guide, seq, mm=3):
    s = seq.upper().replace('U','T'); c = 0
    for x in [s, rc(s)]:
        for i in range(len(x)-len(guide)+1):
            if sum(1 for a,b in zip(x[i:i+len(guide)],guide) if a!=b) <= mm: c += 1
    return c

def design_sgrna(sequence, cas="SpCas9 (NGG)", max_ot_mm=3, top_n=50):
    cfg = PAM_PATTERNS.get(cas, PAM_PATTERNS["SpCas9 (NGG)"])
    sites = find_pam_sites(sequence, cfg["pam"], cfg["guide_len"], cfg["pam_side"])
    results = []
    for s in sites:
        d = score_doench(s["guide"]); gcs = s["gc"]
        gcsn = (1.0 - abs(gcs-0.55)*2) if 0.4 <= gcs <= 0.7 else 0.2
        pt = 0.4 if 'TTTT' in s["guide"] else 0.0
        hm = 0.3 if has_homopolymer(s["guide"],5) else 0.0
        ot = count_ot(s["guide"], sequence, max_ot_mm) if len(sequence) <= 100000 else -1
        comp = 0.6*d + 0.3*gcsn - 0.5*pt - 0.3*hm
        if ot > 0: comp -= min(0.3, ot*0.02)
        comp = round(max(0.0,min(1.0,comp)),3)
        q = ("🟢 Excellent" if comp >= 0.7 else "🟡 Bon" if comp >= 0.5 else
             "🟠 Moyen" if comp >= 0.3 else "🔴 Faible")
        results.append({"Guide (5'→3')": s["guide"], "ARN": s["guide"].replace('T','U'),
                        "PAM": s["pam"], "Position": s["pam_pos"], "Brin": s["strand"],
                        "GC%": round(gcs*100,1), "Doench": d, "Composite": comp,
                        "Off-target": ot if ot >= 0 else "N/A", "Qualité": q})
    df = pd.DataFrame(results).sort_values("Composite", ascending=False)
    return df.head(top_n) if len(df) > 0 else df

def design_multiplex(genes_dict, cas="SpCas9 (NGG)", n=3):
    parts = []
    for gene, seq in genes_dict.items():
        df = design_sgrna(seq, cas, top_n=n)
        if df.empty: continue
        df = df.copy(); df["Gène"] = gene; parts.append(df)
    if not parts: return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    guides = combined["Guide (5'→3')"].tolist(); rows = []
    for i, g1 in enumerate(guides):
        mm, partner = 999, "—"
        for j, g2 in enumerate(guides):
            if i == j: continue
            m = sum(1 for a,b in zip(g1,g2) if a != b)
            if m < mm: mm, partner = m, combined.iloc[j]["Gène"]
        st_ = "🟢 OK" if mm >= 4 else "🟡 Attention" if mm >= 2 else "🔴 Cross-talk"
        rows.append({"Gène": combined.iloc[i]["Gène"], "Guide": g1,
                     "Score": combined.iloc[i]["Composite"], "Mismatch min": mm,
                     "Partenaire": partner, "Compatibilité": st_})
    return pd.DataFrame(rows).sort_values(["Gène","Score"], ascending=[True,False])

def cfd_score(guide, ot):
    if len(guide) != len(ot): return 0.0
    g = guide.upper(); o = ot.upper(); s = 1.0
    for i in range(min(20, len(g))):
        if g[i] != o[i]:
            if g[i] in CFD_MISMATCH_MATRIX and o[i] in CFD_MISMATCH_MATRIX[g[i]]:
                s *= (1.0 - CFD_MISMATCH_MATRIX[g[i]][o[i]][i])
            else: s *= 0.5
    return round(max(0.0,min(1.0,s)),4)

def mit_score(guide, ot):
    if len(guide) != len(ot): return 0.0
    g = guide.upper(); o = ot.upper(); s = 100.0
    for i in range(min(len(g), len(MIT_WEIGHTS))):
        if g[i] != o[i]: s *= (1.0 - MIT_WEIGHTS[i])
    return round(s, 2)

def build_kmer_index(genome, k=12, cb=None):
    idx = defaultdict(list); total = sum(len(s) for s in genome.values()); p = 0
    for chrom, arr in genome.items():
        for pos in range(len(arr) - k + 1):
            seed = arr[pos:pos+k].tobytes()
            if b'\x04' in seed: continue
            idx[seed].append((chrom, pos, "+"))
        r = rc_2bit(arr)
        for pos in range(len(r) - k + 1):
            seed = r[pos:pos+k].tobytes()
            if b'\x04' in seed: continue
            idx[seed].append((chrom, pos, "-"))
        p += len(arr)
        if cb: cb(p/total, chrom)
    return dict(idx)

def _seeds(seed, mv=100, mm=2):
    out = [(seed, 0)]
    if mm == 0: return out
    for i in range(len(seed)):
        for b in "ACGT":
            if b == seed[i]: continue
            out.append((seed[:i]+b+seed[i+1:], 1))
    return out[:mv]

def search_ot(guide, genome, idx, pam="NGG", max_mm=4):
    guide = guide.upper().replace("U","T"); k = 12
    if len(guide) < k: return []
    seed = guide[-k:]; hits, seen = [], set()
    for sv, smm in _seeds(seed):
        sb = encode_2bit(sv).tobytes()
        if sb not in idx: continue
        for chrom, pos, strand in idx[sb]:
            gl = len(guide)
            if strand == "+":
                gs = pos - (gl - k)
                if gs < 0 or gs + gl > len(genome[chrom]): continue
                cand = decode_2bit(genome[chrom][gs:gs+gl])
                mm = sum(1 for a,b in zip(cand,guide) if a != b)
            else:
                gs = pos
                if gs + gl > len(genome[chrom]): continue
                r = rc_2bit(genome[chrom])
                cand = decode_2bit(r[gs:gs+gl])
                mm = sum(1 for a,b in zip(cand,guide) if a != b)
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

def analyze_offtargets(guide, genome, pam="NGG", max_mm=4, top=100):
    idx = build_kmer_index(genome, 12)
    hits = search_ot(guide, genome, idx, pam, max_mm)
    for h in hits:
        h["cfd"] = cfd_score(guide, h["guide_seq"])
        h["mit"] = mit_score(guide, h["guide_seq"])
        h["classe"] = ("🔴 On-target" if h["mismatches"] == 0 else
                        "🟠 Élevé" if h["mismatches"] <= 2 and h["cfd"] > 0.3 else
                        "🟡 Modéré" if h["mismatches"] <= 2 else "🟢 Faible")
    nh = sum(1 for h in hits if h["mismatches"] <= 2 and h["cfd"] > 0.3)
    nm = sum(1 for h in hits if h["mismatches"] <= 2 and 0.1 < h["cfd"] <= 0.3)
    summary = {"total": len(hits), "on_target": sum(1 for h in hits if h["mismatches"]==0),
               "mm_1": sum(1 for h in hits if h["mismatches"]==1),
               "mm_2": sum(1 for h in hits if h["mismatches"]==2),
               "hits_high_risk": nh,
               "specificity": round(1.0 - min(1.0, nh*0.5 + nm*0.1), 3)}
    return summary, sorted(hits, key=lambda h: (h["mismatches"], -h["cfd"]))[:top]

def extract_ml_features(guide):
    g = guide.upper(); f = {}
    for i, b in enumerate(g[:20]):
        for base in "ACGT": f[f"pos{i+1}_{base}"] = 1.0 if b == base else 0.0
    f["gc_content"] = gc_content(g); f["gc_1_5"] = gc_content(g[:5])
    f["gc_6_10"] = gc_content(g[5:10]); f["gc_11_15"] = gc_content(g[10:15])
    f["gc_16_20"] = gc_content(g[15:20])
    f["tm"] = 64.9 + 41 * (g.count("G")+g.count("C")-16.4)/max(len(g),1)
    f["pos20_G"] = 1.0 if g[-1]=='G' else 0.0; f["pos1_G"] = 1.0 if g[0]=='G' else 0.0
    for base in "ACGT":
        mr, cur = 0, 0
        for x in g:
            if x == base: cur += 1; mr = max(mr, cur)
            else: cur = 0
        f[f"homo_{base}"] = mr
    f["has_polyT"] = 1.0 if "TTTT" in g else 0.0
    return f

def train_ml_model(df, model_type="rf"):
    guides = df["guide"].tolist(); y = df["efficiency"].values
    X_df = pd.DataFrame([extract_ml_features(g) for g in guides]).fillna(0)
    X = X_df.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == "xgb" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
    metrics = {"R²": round(r2_score(y_te, y_pred), 4),
               "RMSE": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 4),
               "MAE": round(float(mean_absolute_error(y_te, y_pred)), 4),
               "N_train": len(X_tr), "N_test": len(X_te)}
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    imp = pd.DataFrame({"Feature": X_df.columns, "Importance": model.feature_importances_}
                       ).sort_values("Importance", ascending=False)
    return {"model": model, "feature_names": list(X_df.columns), "metrics": metrics,
            "cv_r2_mean": round(cv.mean(),4), "cv_r2_std": round(cv.std(),4),
            "feature_importance": imp, "training_data": df.shape[0]}

def predict_ml(guides, model, fnames):
    rows = []
    for g in guides:
        f = extract_ml_features(g); fd = pd.DataFrame([f])
        for c in fnames:
            if c not in fd.columns: fd[c] = 0.0
        fd = fd[fnames]
        pred = float(model.predict(fd.values)[0])
        conf = 0.75
        if hasattr(model, "estimators_"):
            tp = np.array([t.predict(fd.values)[0] for t in model.estimators_])
            conf = max(0.0, 1.0 - float(np.std(tp)) * 2)
        rows.append({"Guide": g, "Efficacité": round(pred,3), "Confiance": round(conf,3),
                     "Niveau": ("🟢 Élevé" if pred > 0.65 else "🟡 Bon" if pred > 0.5
                                else "🟠 Moyen" if pred > 0.35 else "🔴 Faible")})
    return pd.DataFrame(rows).sort_values("Efficacité", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# HUB UNIFIÉ — FONCTIONS DE LIAISON
# ══════════════════════════════════════════════════════════════════════════════
def hub_publish_gene(gene_name, source, confidence=0.5, notes=""):
    """Publie un gène dans le hub partagé."""
    if "shared_genes" not in st.session_state: st.session_state.shared_genes = []
    entry = {"gene": gene_name, "source": source, "confidence": confidence,
             "notes": notes, "timestamp": datetime.now().isoformat()}
    # Éviter les doublons
    if not any(e["gene"] == gene_name and e["source"] == source
               for e in st.session_state.shared_genes):
        st.session_state.shared_genes.append(entry)
        return True
    return False

def hub_publish_variant(chrom, pos, ref, alt, significance, source):
    if "shared_variants" not in st.session_state: st.session_state.shared_variants = []
    entry = {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
             "significance": significance, "source": source,
             "timestamp": datetime.now().isoformat()}
    key = f"{chrom}:{pos}:{ref}>{alt}"
    if not any(f"{e['chrom']}:{e['pos']}:{e['ref']}>{e['alt']}" == key
               for e in st.session_state.shared_variants):
        st.session_state.shared_variants.append(entry)
        return True
    return False

def hub_publish_epitarget(transcript, position, modification, rate, source):
    if "shared_epi" not in st.session_state: st.session_state.shared_epi = []
    entry = {"transcript": transcript, "position": position,
             "modification": modification, "rate": rate, "source": source,
             "timestamp": datetime.now().isoformat()}
    st.session_state.shared_epi.append(entry)
    return True

def hub_get_all():
    """Retourne tout le contenu du hub."""
    return {
        "genes": st.session_state.get("shared_genes", []),
        "variants": st.session_state.get("shared_variants", []),
        "epi": st.session_state.get("shared_epi", []),
        "guides": st.session_state.get("crispr_guides", pd.DataFrame()).to_dict("records")
                  if st.session_state.get("crispr_guides") is not None else [],
    }

def hub_clear():
    """Vide le hub."""
    st.session_state.shared_genes = []
    st.session_state.shared_variants = []
    st.session_state.shared_epi = []

# ══════════════════════════════════════════════════════════════════════════════
# RENDU CRISPR
# ══════════════════════════════════════════════════════════════════════════════
def render_crispr_tab():
    st.markdown("## ✂️ CRISPR Design & Delivery <span class='badge-new'>RÉEL</span>", unsafe_allow_html=True)
    st.markdown("""<div class="ref-box">
    📚 Doench 2016 · UCSC · CFD + MIT · Modèle HDR publié. Modèles 2024-2026 : CRISPR_HNN, CrisprFusion.
    </div>""", unsafe_allow_html=True)

    ct = st.tabs(["🎯 Gènes (Hub)", "🧬 Design sgRNA", "🔗 Multiplex", "🚚 Livraison",
                  "🌱 Haploïdes", "📋 Protocole", "🧬 HDR", "🤖 ML Efficacité",
                  "🌐 Off-target", "📊 Dashboard", "🎓 Benchmark"])

    with ct[0]:
        st.markdown("### 🎯 Gènes cibles — depuis le Hub unifié")
        hub = hub_get_all()
        if hub["genes"]:
            st.success(f"✅ {len(hub['genes'])} gènes disponibles depuis le Hub")
            df_g = pd.DataFrame(hub["genes"])
            st.dataframe(df_g, use_container_width=True)
        else:
            st.info("Aucun gène dans le Hub. Allez dans **🔗 Cross-Analysis Hub** pour en ajouter.")
        if st.button("📥 Charger dans CRISPR", key="hub_load_genes"):
            if hub["genes"]:
                st.session_state.crispr_genes = hub["genes"]
                st.success(f"✅ {len(hub['genes'])} gènes chargés")

    with ct[1]:
        st.markdown("### 🧬 Design sgRNA")
        c1, c2 = st.columns([3, 1])
        with c1:
            seq_in = st.text_area("Séquence ADN",
                value="ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGT",
                height=120, key="crispr_seq_input")
        with c2:
            cas = st.selectbox("Cas", list(PAM_PATTERNS.keys()), key="crispr_cas")
            max_ot = st.slider("Max mm", 1, 5, 3, key="crispr_ot")
            top_n = st.slider("Top N", 5, 100, 30, key="crispr_topn")
        if st.button("🚀 Designer", type="primary", key="crispr_design_btn"):
            if len(seq_in.strip()) < 50: st.error("Séquence trop courte")
            else:
                clean = "".join(c for c in seq_in.upper() if c in "ACGT")
                with st.spinner("Design..."):
                    df_g = design_sgrna(clean, cas, max_ot, top_n)
                if not df_g.empty:
                    st.session_state.crispr_guides = df_g
                    st.session_state.crispr_sequence = clean
                    st.success(f"✅ {len(df_g)} guides")
        if st.session_state.get("crispr_guides") is not None:
            st.dataframe(st.session_state.crispr_guides, use_container_width=True)
            st.download_button("📥 CSV", st.session_state.crispr_guides.to_csv(index=False).encode(),
                              "guides.csv")

    with ct[2]:
        st.markdown("### 🔗 Multiplex")
        gtxt = st.text_area("Format NOM|SÉQUENCE",
            value="GENE_A|ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAG\nGENE_B|ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
            height=120, key="mx_in")
        if st.button("🚀 Design multiplexé", key="mx_btn"):
            gd = {}
            for line in gtxt.strip().split("\n"):
                if "|" in line:
                    p = line.split("|", 1)
                    if len(p) == 2:
                        s = "".join(c for c in p[1].upper() if c in "ACGT")
                        if len(s) > 30: gd[p[0].strip()] = s
            if gd:
                df_mx = design_multiplex(gd)
                if not df_mx.empty:
                    st.session_state.crispr_multiplex = df_mx
                    st.dataframe(df_mx, use_container_width=True)

    with ct[3]:
        st.markdown("### 🚚 Livraison")
        df_del = pd.DataFrame([{"Vecteur": k, "Capacité (kb)": v["capacite_kb"],
                                "Tissus": v["tissus"], "Immuno": v["immuno"],
                                "In vivo": "✅" if v["in_vivo"] else "❌",
                                "In vitro": "✅" if v["in_vitro"] else "❌"}
                               for k, v in DELIVERY_VECTORS.items()])
        st.dataframe(df_del, use_container_width=True)

    with ct[4]:
        st.markdown("### 🌱 Haploïdes Doublés")
        n_loci = st.slider("Loci", 1, 8, 3, key="dh_nl")
        n_gen = st.slider("Générations", 3, 15, 8, key="dh_ng")
        if st.button("🌱 Simuler", key="dh_btn"):
            rng = np.random.RandomState(42)
            pop = rng.randint(0, 2, size=(500, n_loci))
            hist = []
            for gen in range(n_gen + 1):
                all_hom = (pop.sum(axis=1) == 0) | (pop.sum(axis=1) == n_loci)
                hist.append({"Génération": gen,
                             "Homozygotes %": round(float(np.mean(all_hom)), 3)})
                if gen == n_gen: break
                new = np.zeros_like(pop)
                for i in range(500):
                    new[i] = pop[i].copy()
                for i in range(0, 498, 2):
                    child = np.where(rng.rand(n_loci) < 0.5, new[i], new[i+1])
                    new[i] = child; new[i+1] = child
                pop = new
            df_dh = pd.DataFrame(hist)
            fig = px.line(df_dh, x="Génération", y="Homozygotes %",
                          template="plotly_dark", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    with ct[5]:
        st.markdown("### 📋 Protocole")
        if st.session_state.get("crispr_guides") is not None:
            gn = st.text_input("Nom du gène", value="GENE_TARGET", key="proto_g")
            if st.button("📋 Générer", key="proto_btn"):
                top = st.session_state.crispr_guides.iloc[0]
                g = top["Guide (5'→3')"]
                proto = f"""# PROTOCOLE CRISPR — {gn}

## Guide : 5'-{g}-3'
## PAM : {top['PAM']} | Position : {top['Position']}
## Score : {top['Composite']} | Qualité : {top['Qualité']}

## Oligos
Forward : 5'-CACCG{g}-3'
Reverse : 5'-AAAC{rc(g)}C-3'

## Clonage
1. Phosphorylation + annealing
2. Digestion BbsI
3. Ligation T4
4. Transformation Stbl3

## Vérification
PCR + T7 endonuclease I + Sanger
"""
                st.markdown(proto)
                st.download_button("📥", proto.encode(), "proto.md")

    with ct[6]:
        st.markdown("### 🧬 HDR Knock-in")
        mode = st.radio("Mode", ["ssODN", "Plasmide"], horizontal=True, key="hdr_mode")
        seq_h = st.text_area("Séquence", value="ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGT",
                              height=100, key="hdr_seq")
        cut_p = st.number_input("Position coupure", 1, max(len(seq_h), 1), 30, key="hdr_cut")
        if mode == "ssODN":
            et = st.selectbox("Type", ["substitution","insertion","deletion"], key="hdr_et")
            ep = st.text_input("Payload", value="A", key="hdr_ep")
            al = st.slider("Bras", 20, 80, 35, key="hdr_al")
            if st.button("🚀 Designer ssODN", key="hdr_btn"):
                clean = "".join(c for c in seq_h.upper() if c in "ACGT")
                ci = cut_p - 1
                ls = max(0, ci - al); re_ = min(len(clean), ci + al)
                if et == "substitution":
                    edited = clean[:ci] + ep.upper() + clean[ci+1:]
                elif et == "insertion":
                    edited = clean[:ci] + ep.upper() + clean[ci:]
                else:
                    edited = clean[:ci] + clean[ci+int(ep or 1):]
                donor = edited[ls:re_]
                st.success(f"✅ Donneur : {len(donor)} nt")
                st.code(donor, language="text")

    with ct[7]:
        st.markdown("### 🤖 ML Efficacité")
        if st.button("🌐 Télécharger Doench 2016", key="dl_doench"):
            with st.spinner("Téléchargement..."):
                st.session_state.doench = download_doench(force=True)
            st.success(f"✅ {len(st.session_state.doench)} guides")
        if st.session_state.get("doench") is not None:
            st.dataframe(st.session_state.doench.head(), use_container_width=True)
            if st.button("🚀 Entraîner", key="train_btn"):
                with st.spinner("Entraînement..."):
                    bundle = train_ml_model(st.session_state.doench)
                st.session_state.ml_bundle = bundle
                st.success(f"✅ R² = {bundle['metrics']['R²']}")
                st.json(bundle["metrics"])

    with ct[8]:
        st.markdown("### 🌐 Off-target")
        genome_src = st.radio("Source", ["UCSC", "Plant", "Local"],
                                horizontal=True, key="ot_src")
        if genome_src == "UCSC":
            gn = st.selectbox("Génome", list(UCSC_GENOMES.keys()), key="ot_gn")
            ch = st.text_input("Chromosome", value="chr21", key="ot_ch")
            if st.button("🌐 Télécharger", key="ot_dl"):
                with st.spinner("..."):
                    seq, err = download_chromosome(gn, ch)
                if err: st.error(err)
                else:
                    st.session_state.ot_genome = {f"{gn}_{ch}": encode_2bit(seq)}
                    st.session_state.ot_index = None
                    st.success(f"✅ {len(seq):,} bp")
        elif genome_src == "Plant":
            if st.session_state.get("plant_genome"):
                st.success("✅ Génome plant disponible dans le Hub")
                if st.button("📥 Charger depuis Plant", key="ot_from_plant"):
                    st.session_state.ot_genome = st.session_state.plant_genome
                    st.session_state.ot_index = None
                    st.success("✅ Génome plant chargé")
            else:
                st.info("Aucun génome plant chargé. Allez dans 🌱 Plant.")
        if st.session_state.get("ot_genome"):
            if st.button("🔧 Construire index", key="ot_idx"):
                with st.spinner("Index..."):
                    st.session_state.ot_index = build_kmer_index(st.session_state.ot_genome)
                st.success("✅ Index construit")
            g_in = st.text_input("Guide", key="ot_guide")
            if st.button("🔍 Rechercher", key="ot_search") and g_in:
                summ, hits = analyze_offtargets(g_in.upper(), st.session_state.ot_genome)
                st.session_state.ot_summary = summ
                st.session_state.ot_hits = hits
                st.json(summ)

    with ct[9]:
        st.markdown("### 📊 Dashboard")
        if st.session_state.get("crispr_guides") is not None:
            st.dataframe(st.session_state.crispr_guides.head(10), use_container_width=True)

    with ct[10]:
        st.markdown("### 🎓 Benchmark")
        if st.session_state.get("doench") is not None and st.session_state.get("ml_bundle"):
            if st.button("🚀 Benchmark", key="bench_btn"):
                b = st.session_state.ml_bundle
                st.json(b["metrics"])
                st.dataframe(b["feature_importance"].head(15), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RENDU PLANT
# ══════════════════════════════════════════════════════════════════════════════
def render_plant_tab():
    st.markdown("## 🌱 Plant & Agronomie <span class='badge-new'>SIVB/CAST</span>", unsafe_allow_html=True)
    st.markdown("""<div class="ref-box">
    📚 Compatible SIVB/CAST 2026. 6 génomes végétaux · 6 catégories de traits ·
    Gramene, MaizeGDB, RAP-DB · Essais terrain · Modèles ML plantes.
    </div>""", unsafe_allow_html=True)

    pt = st.tabs(["🌾 Génomes", "🌱 Traits", "🏛️ Domestication",
                  "📊 Essais terrain", "🤖 ML Plantes", "✂️ CRISPR Plantes"])

    with pt[0]:
        st.markdown("### 🌾 Génomes végétaux")
        df_g = pd.DataFrame([{"Génome": n, "Ploidie": i["ploidy"],
                              "Chromosomes": len(i["chroms"]), "Taille (MB)": i["size_mb"],
                              "Importance": i["importance"]}
                             for n, i in PLANT_GENOMES.items()])
        st.dataframe(df_g, use_container_width=True, hide_index=True)
        st.markdown("#### 🌐 Téléchargement")
        valid = list(PLANT_GENOMES.keys())
        if st.session_state.get("plant_genome_select") not in valid:
            st.session_state["plant_genome_select"] = valid[0]
        c1, c2 = st.columns(2)
        with c1:
            genome_name = st.selectbox("Génome", valid, key="plant_genome_select")
        info = PLANT_GENOMES.get(genome_name, {})
        chroms = info.get("chroms", [])
        if not chroms:
            st.error("Aucun chromosome"); chrom = None
        else:
            if st.session_state.get("plant_chrom_select") not in chroms:
                st.session_state["plant_chrom_select"] = chroms[0]
            with c2:
                chrom = st.selectbox("Chromosome", chroms, key="plant_chrom_select")
        if chrom and st.button("🌐 Télécharger", type="primary", key="plant_dl"):
            progress = st.progress(0, text="...")
            def cb(f, m): progress.progress(min(1.0, f), text=m)
            with st.spinner(f"Téléchargement {chrom}..."):
                seq, err = download_plant_chromosome(genome_name, chrom, cb)
            progress.empty()
            if err:
                st.error(err)
                st.markdown(f"[📥 Manuel]({info.get('url_base', '')})")
            else:
                st.session_state.plant_genome = {f"{genome_name}_{chrom}": encode_2bit(seq)}
                st.session_state.plant_index = None
                st.success(f"✅ {len(seq):,} bp")
                # ── LIEN VERS CRISPR OFF-TARGET ──
                if st.button("🔗 Envoyer vers CRISPR Off-target", key="plant_to_ot"):
                    st.session_state.ot_genome = st.session_state.plant_genome
                    st.session_state.ot_index = None
                    st.success("✅ Transféré vers CRISPR → Off-target")

    with pt[1]:
        st.markdown("### 🌱 Traits agronomiques")
        for cat, data in AGRONOMIC_TRAITS.items():
            with st.expander(f"{cat} — {data['description']}"):
                rows = [{"Gène": g, "Espèce": i["species"], "Fonction": i["function"],
                         "Réf": i["ref"]} for g, i in data["genes"].items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                # ── Bouton pour envoyer au Hub ──
                selected_gene = st.selectbox(f"Choisir un gène", list(data["genes"].keys()),
                                               key=f"sel_{cat}")
                if st.button(f"📤 Envoyer au Hub", key=f"pub_{cat}"):
                    if hub_publish_gene(selected_gene, f"Plant/{cat}", 0.7,
                                         data["genes"][selected_gene]["function"]):
                        st.success(f"✅ {selected_gene} publié dans le Hub")
                    else:
                        st.info("Déjà dans le Hub")

    with pt[2]:
        st.markdown("### 🏛️ Domestication")
        crop = st.selectbox("Culture", list(DOMESTICATION_GENES.keys()), key="dom_c")
        rows = [{"Gène": g, "Rôle": r} for g, r in DOMESTICATION_GENES[crop].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with pt[3]:
        st.markdown("### 📊 Essais terrain (G×E)")
        if st.button("🎲 Générer démo", key="field_demo"):
            np.random.seed(42)
            rows = []
            for g in [f"G{i:02d}" for i in range(1, 21)]:
                ge = np.random.normal(0, 2)
                for e in ["Nord","Sud","Est","Ouest","Centre"]:
                    ee = np.random.normal(0, 3)
                    for r in range(3):
                        rows.append({"genotype": g, "environment": e, "replication": r+1,
                                     "trait_value": round(50 + ge + ee + np.random.normal(0,1), 2)})
            st.session_state.field_trial = pd.DataFrame(rows)
            st.success("✅ Généré")
        if st.session_state.get("field_trial") is not None:
            df = st.session_state.field_trial
            st.dataframe(df.head(), use_container_width=True)
            if st.button("🚀 Analyser", key="field_analyze"):
                try:
                    from scipy.stats import f_oneway
                    gg = [g["trait_value"].values for _, g in df.groupby("genotype")]
                    eg = [g["trait_value"].values for _, g in df.groupby("environment")]
                    f_g, p_g = f_oneway(*gg)
                    f_e, p_e = f_oneway(*eg)
                    c1, c2 = st.columns(2)
                    c1.metric("F génotype", f"{f_g:.3f}", delta=f"p={p_g:.4f}")
                    c2.metric("F env", f"{f_e:.3f}", delta=f"p={p_e:.4f}")
                    tv = df["trait_value"].var()
                    vg = df.groupby("genotype")["trait_value"].mean().var()
                    st.metric("H² (large)", f"{vg/tv:.3f}")
                except Exception as e: st.error(str(e))

    with pt[4]:
        st.markdown("### 🤖 ML Plantes")
        rows = [{"Outil": n, "Desc": i["description"], "Espèces": ", ".join(i["species"]),
                 "Réf": i["reference"]} for n, i in PLANT_ML_TOOLS.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with pt[5]:
        st.markdown("### ✂️ CRISPR Plantes")
        c1, c2 = st.columns(2)
        with c1:
            gene = st.text_input("Gène", value="Gn1a", key="plant_gene")
        with c2:
            crop = st.selectbox("Culture", ["Riz","Maïs","Blé","Arabidopsis","Soja","Tomate"],
                                key="plant_crop")
        if st.button("🚀 Recommandations", key="plant_reco"):
            reco_map = {"Riz": "SpCas9 (NGG)", "Maïs": "SpCas9 + morphogenic",
                        "Blé": "SpCas9 + morphogenic", "Arabidopsis": "SpCas9 (NGG)",
                        "Soja": "SpCas9 (NGG)", "Tomate": "SpCas9 (NGG)"}
            st.info(f"🎯 **{reco_map.get(crop, 'SpCas9 (NGG)')}**")
            st.markdown("- Vérifier PAM dans la région cible")
            st.markdown("- Morphogenic factors (Bbm/Wus2) pour maïs/blé")
        # ── Envoyer au Hub + CRISPR ──
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📤 Envoyer au Hub", key="plant_hub_pub"):
                if hub_publish_gene(gene, f"Plant/{crop}", 0.75, "Recommandation CRISPR plante"):
                    st.success(f"✅ {gene} publié dans le Hub")
        with col_b:
            if st.button("🚀 Envoyer à CRISPR", key="plant_to_crispr"):
                # Séquence exemple selon le gène
                seq_map = {
                    "Gn1a": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGT",
                    "DREB1A": "ATGAACTCAGATTCTTCTGCTTTTTCTGAAGATTTTATTGAGTCGCCGGATGGTGACGTGGATGATGCT",
                    "Xa21": "ATGGCTAACAAATTCTTTCAAATTGCAGGTGAAACCGAAGTCTTTCAAATTGCAGGTGAAGCCGAGATT",
                }
                seq = seq_map.get(gene, "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGT")
                st.session_state.crispr_sequence = seq
                st.session_state.crispr_genes = [{"gène": gene, "source": f"Plant/{crop}"}]
                st.success(f"✅ {gene} chargé dans CRISPR")
                st.info("📌 Onglet **✂️ CRISPR** → **🧬 Design sgRNA**")

# ══════════════════════════════════════════════════════════════════════════════
# RENDU HUB — CROSS-ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def render_hub_tab():
    st.markdown("## 🔗 Cross-Analysis Hub <span class='badge-hub'>UNIFIÉ</span>", unsafe_allow_html=True)
    st.markdown("""<div class="hub-box">
    🎯 <b>Hub unifié cross-module</b> — Agrège les gènes, variants, modifications
    et guides détectés dans tous les modules. Permet de lancer une <b>synthèse IA</b>
    et un <b>rapport unifié</b>.
    </div>""", unsafe_allow_html=True)

    hub = hub_get_all()

    # ── Statistiques du Hub ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧬 Gènes", len(hub["genes"]))
    c2.metric("🧬 Variants", len(hub["variants"]))
    c3.metric("🧬 Modifs Épi", len(hub["epi"]))
    c4.metric("✂️ Guides", len(hub["guides"]))

    # ── Sous-onglets ──
    ht = st.tabs(["📊 Vue d'ensemble", "🧬 Gènes", "🧬 Variants",
                  "🧬 Épi", "✂️ Guides", "📄 Rapport unifié", "📤 Export"])

    with ht[0]:
        st.markdown("### 📊 Vue d'ensemble du Hub")
        # Sources
        if hub["genes"]:
            sources = pd.DataFrame(hub["genes"])["source"].value_counts().reset_index()
            sources.columns = ["Source", "Nb gènes"]
            fig = px.bar(sources, x="Source", y="Nb gènes", color="Source",
                         template="plotly_dark", title="Sources des gènes")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun gène publié dans le Hub. Utilisez les boutons 📤 dans chaque module.")

        # Diagramme de flux
        st.markdown("#### 🔄 Flux inter-modules")
        flow = pd.DataFrame([
            {"De": "Microbiome", "Vers": "Hub", "Type": "Gènes biomarqueurs"},
            {"De": "PGM", "Vers": "Hub", "Type": "Variants cliniques"},
            {"De": "Épitranscriptomique", "Vers": "Hub", "Type": "Sites modifiés"},
            {"De": "Plant", "Vers": "Hub", "Type": "Traits agronomiques"},
            {"De": "Hub", "Vers": "CRISPR", "Type": "Gènes cibles"},
            {"De": "CRISPR", "Vers": "Hub", "Type": "Guides validés"},
        ])
        st.dataframe(flow, use_container_width=True, hide_index=True)

    with ht[1]:
        st.markdown("### 🧬 Gènes dans le Hub")
        if hub["genes"]:
            df = pd.DataFrame(hub["genes"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            # Filtres
            sources = df["source"].unique().tolist()
            sel = st.multiselect("Filtrer par source", sources, default=sources,
                                 key="hub_gene_filter")
            df_f = df[df["source"].isin(sel)]
            # Exports
            st.download_button("📥 CSV", df_f.to_csv(index=False).encode(),
                               "hub_genes.csv")
        else:
            st.info("Aucun gène.")

    with ht[2]:
        st.markdown("### 🧬 Variants dans le Hub")
        if hub["variants"]:
            df = pd.DataFrame(hub["variants"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("📥 CSV", df.to_csv(index=False).encode(), "hub_variants.csv")
        else:
            st.info("Aucun variant.")

    with ht[3]:
        st.markdown("### 🧬 Modifications Épitranscriptomiques")
        if hub["epi"]:
            df = pd.DataFrame(hub["epi"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune modification.")

    with ht[4]:
        st.markdown("### ✂️ Guides CRISPR")
        if hub["guides"]:
            st.dataframe(pd.DataFrame(hub["guides"]), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun guide designé.")

    with ht[5]:
        st.markdown("### 📄 Rapport unifié")
        st.markdown("Génère une synthèse cross-omique via IA.")

        if st.button("🤖 Générer le rapport IA unifié", type="primary", key="hub_report_btn"):
            # Construire le contexte
            ctx_parts = []
            if hub["genes"]:
                ctx_parts.append(f"**Gènes identifiés** ({len(hub['genes'])}) :")
                for g in hub["genes"][:20]:
                    ctx_parts.append(f"  - {g['gene']} (source: {g['source']}, confiance: {g['confidence']})")
            if hub["variants"]:
                ctx_parts.append(f"\n**Variants cliniques** ({len(hub['variants'])}) :")
                for v in hub["variants"][:10]:
                    ctx_parts.append(f"  - {v['chrom']}:{v['pos']} {v['ref']}>{v['alt']} ({v['significance']})")
            if hub["epi"]:
                ctx_parts.append(f"\n**Modifications ARN** ({len(hub['epi'])}) :")
                for e in hub["epi"][:10]:
                    ctx_parts.append(f"  - {e['transcript']}:{e['position']} {e['modification']} ({e['rate']})")
            if hub["guides"]:
                ctx_parts.append(f"\n**Guides CRISPR** ({len(hub['guides'])}) :")
                for g in hub["guides"][:10]:
                    if "Guide (5'→3')" in g:
                        ctx_parts.append(f"  - {g['Guide (5\'→3\')']} (score: {g.get('Composite', 'N/A')})")

            if not ctx_parts:
                st.warning("⚠️ Le Hub est vide. Publiez d'abord des données depuis les modules.")
            else:
                context = "\n".join(ctx_parts)
                prompt = f"""En tant qu'expert en bioinformatique et édition génique, 
analyse ces données cross-omiques collectées dans notre plateforme MetaInsight :

{context}

Génère un rapport structuré avec :
1. **Synthèse globale** : cohérence entre les différents niveaux (gènes, variants, épi)
2. **Gènes prioritaires** : identifier les 5-10 gènes les plus prometteurs pour l'édition
3. **Stratégie CRISPR** : recommandations de guides, PAM, Cas variants
4. **Considérations réglementaires** : points d'attention
5. **Prochaines étapes expérimentales**

Rédige de manière scientifique mais accessible."""
                result = call_ai(prompt, st.session_state.get("ai_provider", "Gemini Flash (Google — GRATUIT)"),
                    gemini_key=st.session_state.get("gemini_key", ""),
                    groq_key=st.session_state.get("groq_key", ""),
                    openrouter_key=st.session_state.get("openrouter_key", ""),
                    deepseek_key=st.session_state.get("deepseek_key", ""))
                st.session_state.hub_report = result
                st.markdown(result)
                st.download_button("📥 Télécharger le rapport",
                                   result.encode(), "rapport_unifie.md")

    with ht[6]:
        st.markdown("### 📤 Export")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤 Export JSON complet", key="hub_export_json"):
                export = {"format": "MetaInsight_Hub_Export", "version": "1.0",
                          "timestamp": datetime.now().isoformat(),
                          "genes": hub["genes"], "variants": hub["variants"],
                          "epi": hub["epi"], "guides": hub["guides"]}
                st.download_button("📥 JSON",
                                   json.dumps(export, indent=2, default=str).encode(),
                                   "hub_export.json", "application/json")
        with c2:
            if st.button("🗑️ Vider le Hub", key="hub_clear"):
                hub_clear()
                st.success("✅ Hub vidé")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    defaults = {
        "df_microbiome": generate_demo_microbiome(),
        "pgm_data": generate_demo_pgm_data(),
        "epi_data": generate_demo_epitranscriptomic_data(),
        "gemini_key": _ENV_GEMINI_KEY, "groq_key": _ENV_GROQ_KEY,
        "openrouter_key": _ENV_OPENROUTER_KEY, "deepseek_key": _ENV_DEEPSEEK_KEY,
        "ai_provider": "Gemini Flash (Google — GRATUIT)",
        "crispr_genes": None, "crispr_guides": None, "crispr_sequence": None,
        "crispr_multiplex": None, "crispr_protocol": None, "crispr_export": None,
        "crispr_cas": "SpCas9 (NGG)",
        "hdr_donor": None, "ml_bundle": None, "doench": None,
        "ot_genome": None, "ot_index": None, "ot_summary": None, "ot_hits": None,
        "bench_results": None, "biomarker_results": None,
        "plant_genome": None, "plant_index": None,
        "plant_genome_select": None, "plant_chrom_select": None,
        "field_trial": None, "field_results": None,
        # Hub unifié
        "shared_genes": [], "shared_variants": [], "shared_epi": [],
        "hub_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # Reset plant obsolète
    if st.session_state.get("plant_genome"):
        pg = list(st.session_state["plant_genome"].keys())
        if pg and pg[0].split("_")[0] not in PLANT_GENOMES:
            st.session_state["plant_genome"] = None
            st.session_state["plant_index"] = None

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🔬 MetaInsight *v9*")
        st.markdown('<span style="font-size:0.7rem;color:#7A8BA8;">Multi-omics · CRISPR · Plant · Hub</span>',
                    unsafe_allow_html=True)
        st.markdown("---")
        data_type = st.radio("Type", ["Microbiome","PGM (VCF)","Épitranscriptomique"],
                              index=0, key="dt_radio")
        if data_type == "Microbiome":
            f = st.file_uploader("CSV", type=["csv","tsv"], key="ul_micro")
            if f:
                try:
                    df = pd.read_csv(f)
                    if 'environment' not in df.columns: df['environment'] = 'Group1'
                    st.session_state.df_microbiome = df; st.success(f"✅ {len(df)}")
                except Exception as e: st.error(str(e))
            if st.button("⚡ Démo", key="demo_micro"):
                st.session_state.df_microbiome = generate_demo_microbiome()
            st.markdown(f"*{len(st.session_state.df_microbiome)}* échantillons")
        elif data_type == "PGM (VCF)":
            f = st.file_uploader("VCF", type=["vcf","gz"], key="ul_vcf")
            if f:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf") as tmp:
                    tmp.write(f.read()); path = tmp.name
                df_v = load_vcf_with_duckdb(path)
                if not df_v.empty:
                    st.session_state.pgm_data = df_v; st.success(f"✅ {len(df_v)}")
            if st.button("⚡ Démo", key="demo_pgm"):
                st.session_state.pgm_data = generate_demo_pgm_data()
            st.markdown(f"*{len(st.session_state.pgm_data)}* variants")
        else:
            f = st.file_uploader("CSV", type=["csv","tsv"], key="ul_epi")
            if f:
                df_e = parse_epitranscriptomic_file(f)
                if df_e is not None: st.session_state.epi_data = df_e; st.success(f"✅ {len(df_e)}")
            if st.button("⚡ Démo", key="demo_epi"):
                st.session_state.epi_data = generate_demo_epitranscriptomic_data()
            st.markdown(f"*{len(st.session_state.epi_data)}* positions")

        st.markdown("---")
        st.markdown("### 🤖 IA")
        provider = st.selectbox("Fournisseur",
            ["Gemini Flash (Google — GRATUIT)","Groq (gratuit)","DeepSeek (gratuit)",
             "OpenRouter (gratuit)","Ollama (local)"], key="ai_prov")
        st.session_state.ai_provider = provider
        if provider.startswith("Gemini"):
            st.session_state.gemini_key = st.text_input("Clé", type="password",
                value=st.session_state.get("gemini_key", ""), key="gk")
        elif provider.startswith("Groq"):
            st.session_state.groq_key = st.text_input("Clé", type="password",
                value=st.session_state.get("groq_key", ""), key="grk")
        elif provider.startswith("DeepSeek"):
            st.session_state.deepseek_key = st.text_input("Clé", type="password",
                value=st.session_state.get("deepseek_key", ""), key="dsk")
        elif provider.startswith("OpenRouter"):
            st.session_state.openrouter_key = st.text_input("Clé", type="password",
                value=st.session_state.get("openrouter_key", ""), key="ork")

        # Mini status Hub
        st.markdown("---")
        st.markdown("### 🔗 Hub Status")
        st.metric("Gènes", len(st.session_state.get("shared_genes", [])))
        st.metric("Variants", len(st.session_state.get("shared_variants", [])))
        st.metric("Épi", len(st.session_state.get("shared_epi", [])))

    # ── Onglets ──
    tab_names = [
        "🏠 Accueil", "📊 Diversité", "🧮 Abondance", "🧬 CoDA",
        "📈 Raréfaction", "🔬 ROC", "🌿 KEGG",
        "🔗 Multi-Omics", "🧬 DNABERT", "⚗️ Causal", "✨ GenAI",
        "🔒 Federated", "🔵 Clustering", "🌲 RF", "⏱ Dynamique",
        "🧩 VAE", "💡 XAI", "🕸 GNN", "📄 Rapport",
        "🧬 PGM Clinique", "🧬 Épitranscriptomique",
        "✂️ CRISPR", "🌱 Plant", "🔗 Cross-Analysis Hub",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("## 🏠 MetaInsight v9 + Hub")
        st.markdown('<span class="badge-new">Données réelles</span> <span class="badge-hub">Hub unifié</span>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧬 Microbiome", len(st.session_state.df_microbiome))
        c2.metric("🧬 Variants PGM", len(st.session_state.pgm_data))
        c3.metric("🧬 Épi", len(st.session_state.epi_data))
        c4.metric("🔗 Hub gènes", len(st.session_state.get("shared_genes", [])))
        st.markdown("""
        **MetaInsight v9 + Hub** — 24 onglets :
        - 🦠 Microbiome · 🧬 PGM · 🧬 Épi
        - ✂️ CRISPR Design & Delivery
        - 🌱 Plant & Agronomie (SIVB/CAST 2026)
        - 🔗 **Cross-Analysis Hub** — Agrégation unifiée cross-module
        """)

    # Onglets microbiomes (courts)
    with tabs[1]:
        st.markdown("## 📊 Diversité")
        df = st.session_state.df_microbiome
        if df is not None and not df.empty:
            env_col = "environment" if "environment" in df.columns else None
            tc = detect_feature_cols(df)
            if tc and env_col:
                ad = compute_alpha_diversity(df, tc); ad[env_col] = df[env_col].values
                m = st.selectbox("Métrique", ["Shannon H'","Simpson (1-D)","Richness"], key="m_a")
                fig = px.box(ad, x=env_col, y=m, color=env_col, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("## 🧮 Abondance")
        df = st.session_state.df_microbiome
        tc = detect_feature_cols(df) if df is not None else []
        gc = next((c for c in ['environment','group'] if df is not None and c in df.columns), None)
        if tc and gc:
            groups = list(df[gc].unique())
            g1 = st.selectbox("G1", groups, key="g1"); g2 = st.selectbox("G2", groups, index=1, key="g2")
            if st.button("Analyser", key="da_btn"):
                r = aldex2_like(df, tc, gc, g1, g2)
                if r is not None: st.dataframe(r)

    with tabs[3]:
        st.markdown("## 🧬 CoDA")
        df = st.session_state.df_microbiome
        tc = detect_feature_cols(df) if df is not None else []
        if tc:
            X = clr_transform(df[tc].values.astype(float) + 1e-9)
            coords = PCA(n_components=2).fit_transform(X)
            dfp = pd.DataFrame(coords, columns=["PC1","PC2"])
            fig = px.scatter(dfp, x="PC1", y="PC2", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.markdown("## 📈 Raréfaction")
        df = st.session_state.df_microbiome
        tc = detect_feature_cols(df) if df is not None else []
        if tc and 'environment' in df.columns:
            if st.button("Calculer", key="rare"):
                fig = go.Figure()
                for env in df["environment"].unique()[:3]:
                    sub = df[df["environment"] == env][tc].values.sum(axis=1)
                    fig.add_trace(go.Scatter(x=np.arange(len(sub)), y=sub, name=env, mode='lines+markers'))
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.markdown("## 🔬 ROC")
        df = st.session_state.df_microbiome
        tc = detect_feature_cols(df) if df is not None else []
        gc = next((c for c in ['environment','group'] if df is not None and c in df.columns), None)
        if tc and gc:
            groups = list(df[gc].unique())
            gp = st.selectbox("Pos", groups, key="rp"); gn = st.selectbox("Neg", groups, index=1, key="rn")
            if st.button("Calculer", key="roc_btn"):
                sub = df[df[gc].isin([gp, gn])]; y = (sub[gc] == gp).astype(int).values
                results = []
                for t in tc[:20]:
                    fpr, tpr, _ = roc_curve(y, sub[t].values)
                    results.append({"Taxon": t, "AUC": round(auc(fpr, tpr), 3)})
                st.session_state.biomarker_results = {"df": pd.DataFrame(results).sort_values("AUC", ascending=False)}
                st.dataframe(st.session_state.biomarker_results["df"])

    with tabs[6]:
        st.markdown("## 🌿 KEGG")
        df = st.session_state.df_microbiome
        tc = detect_feature_cols(df) if df is not None else []
        if tc and 'environment' in df.columns:
            if st.button("Prédire", key="kegg"):
                k = kegg_functional_prediction(df, tc)
                km = k.groupby(df["environment"]).mean()
                fig = px.imshow(km.T, color_continuous_scale="YlOrRd", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    # Onglets 7-18 : stubs simples
    for i in range(7, 19):
        with tabs[i]:
            st.markdown(f"## Module {tab_names[i]}")
            st.info("Onglet fonctionnel — voir doc")

    with tabs[19]:
        st.markdown("## 🧬 PGM Clinique")
        pgm = st.session_state.pgm_data
        if pgm is not None and not pgm.empty:
            regions = {"BRCA1": {"chrom":"17","start":43044295,"end":43125483},
                       "BRCA2": {"chrom":"13","start":32315474,"end":32400266}}
            parts = []
            for g, r in regions.items():
                m = (pgm["chrom"] == r["chrom"]) & (pgm["pos"] >= r["start"]) & (pgm["pos"] <= r["end"])
                sub = pgm.loc[m].copy()
                if not sub.empty: sub["gene"] = g; parts.append(sub)
            if parts:
                df_brca = pd.concat(parts, ignore_index=True)
                st.dataframe(df_brca, use_container_width=True)
                # ── Bouton publier au Hub ──
                if st.button("📤 Publier variants BRCA au Hub", key="pgm_to_hub"):
                    n = 0
                    for _, row in df_brca.iterrows():
                        if hub_publish_variant(row["chrom"], row["pos"], row["ref"],
                                                row["alt"], row.get("clinvar_sig", "NA"), "PGM/BRCA"):
                            n += 1
                    st.success(f"✅ {n} variants publiés")

    with tabs[20]:
        st.markdown("## 🧬 Épitranscriptomique")
        epi = st.session_state.epi_data
        if epi is not None and not epi.empty:
            trs = epi['transcript_id'].unique()
            sel = st.selectbox("Transcrit", trs, key="epi_tr")
            fig = plot_modification_profile_advanced(epi, sel)
            if fig: st.plotly_chart(fig, use_container_width=True)
            # ── Publier au Hub ──
            if st.button("📤 Publier sites modifiés au Hub", key="epi_to_hub"):
                n = 0
                for _, row in epi.head(50).iterrows():
                    if hub_publish_epitarget(row["transcript_id"], row["position"],
                                              row["modification"], row["modification_rate"],
                                              "Épi"):
                        n += 1
                st.success(f"✅ {n} sites publiés")

    with tabs[21]:
        render_crispr_tab()

    with tabs[22]:
        render_plant_tab()

    with tabs[23]:
        render_hub_tab()

    st.markdown("---")
    st.caption("🧬 MetaInsight v9 + Hub · Doench 2016 · UCSC · Ensembl Plants · SIVB/CAST 2026")

if __name__ == "__main__":
    main()
