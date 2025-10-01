import os, re, ast
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ======================= Config =======================
NUMERIC_COLS     = ["price", "size_sqm", "rooms", "floor", "total_floors", "price_per_sqm"]
CATEGORICAL_COLS = ["apartment_style", "neighborhood", "city_group", "city"]
BOOLEAN_COLS     = ["elevator", "wheelchair_access", "tornado_ac", "multi_bolt_doors", "air_conditioning", "bars"]
EMBEDDING_COL    = "description_embedding"  # optional

APP_DIR  = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "Data"
DATA_DIR.mkdir(exist_ok=True)

# Preferred CSVs: clean -> raw -> sample
_CANDIDATES = [
    DATA_DIR / "clean_realestate.csv",
    DATA_DIR / "realestate.csv",
    DATA_DIR / "sample_realestate.csv",
]

def _pick_existing_csv() -> str:
    for p in _CANDIDATES:
        if p.exists():
            return str(p)
    return ""  # none found

DEFAULT_RESOLVED_CSV = _pick_existing_csv()

st.set_page_config(page_title="🏠 Apartment Recommender", layout="wide")
st.title("🏠 Apartment Recommender — Similar listings, user profile & chat")

# ======================= Small helpers =======================
def to_numeric_flex(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip()
    if s in {"", "NA", "NaN", "nan", "None", "Not specified", "No price", "Unknown"}: return np.nan
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s) if s not in {"", "-", ".", "-.", ".-"} else np.nan
    except Exception:
        return np.nan

def make_ohe():
    # sklearn compatibility (sparse_output introduced in 1.2)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def parse_embedding_cell(x):
    """Supports list/tuple/ndarray, Python '[...]' string, or CSV '0.1,0.2,...'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr if arr.size else None
    if isinstance(x, str):
        s = x.strip()
        # Try Python literal list first
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                arr = np.asarray(v, dtype=float)
                return arr if arr.size else None
        except Exception:
            pass
        # Fallback: CSV
        try:
            parts = [p for p in s.replace("[", "").replace("]", "").split(",") if p.strip() != ""]
            arr = np.asarray([float(p) for p in parts], dtype=float)
            return arr if arr.size else None
        except Exception:
            return None
    return None

def build_embedding_block(df, col=EMBEDDING_COL):
    """Return CSR matrix of L2-normalized embeddings with unified dimension."""
    if col not in df.columns:
        return None, 0
    embs = df[col].apply(parse_embedding_cell)
    nonnull = embs.dropna()
    if nonnull.empty:
        return None, 0
    dims = nonnull.apply(lambda a: a.shape[0])
    dim = int(dims.mode().iloc[0])

    def fix_dim(a):
        if a is None:
            return np.zeros(dim, dtype=float)
        if a.shape[0] == dim:
            return a
        if a.shape[0] > dim:
            return a[:dim]
        out = np.zeros(dim, dtype=float)
        out[:a.shape[0]] = a
        return out

    M = np.vstack(embs.apply(fix_dim).values).astype(float)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M = M / norms
    return csr_matrix(M), dim

# ======================= Sidebar: load + weights =======================
with st.sidebar:
    st.header("Load data & weights")

    csv_path = st.text_input("CSV path (clean_realestate.csv / realestate.csv / sample_realestate.csv)",
                             value=DEFAULT_RESOLVED_CSV)

    try:
        df = pd.read_csv(csv_path) if csv_path else None
        if df is None:
            st.error("No CSV found. Place one under Data/ (e.g., clean_realestate.csv) or provide a path above.")
            st.stop()
    except Exception as e:
        st.error(f"Can't read CSV:\n{csv_path}\n\n{e}")
        st.stop()

    # Ensure item_id
    if "item_id" not in df.columns:
        if "ad_url" in df.columns:
            df["item_id"] = df["ad_url"].astype(str)
            dup = df["item_id"].duplicated(keep=False)
            if dup.any():
                df.loc[dup, "item_id"] = (
                    df.loc[dup].groupby("item_id").cumcount().astype(str) + "_" + df.loc[dup, "item_id"]
                )
        else:
            df.insert(0, "item_id", np.arange(len(df), dtype=int))

    st.success(f"Loaded {len(df):,} rows")

    st.subheader("Feature Weights")
    w_num  = st.slider("Numeric weight",      0.0, 2.0, 1.0, step=0.05)
    w_cat  = st.slider("Categorical weight",  0.0, 2.0, 1.0, step=0.05)
    w_bool = st.slider("Boolean weight",      0.0, 2.0, 1.0, step=0.05)
    w_emb  = st.slider("Embedding weight",    0.0, 2.0, 1.0, step=0.05)

# ======================= Vector blocks =======================
@st.cache_resource(show_spinner=False)
def build_blocks(df_in: pd.DataFrame, weights):
    df = df_in.copy()
    w_num, w_cat, w_bool, w_emb = weights

    # One-time numeric price
    if "price" in df.columns and "price_num" not in df.columns:
        df["price_num"] = df["price"].apply(to_numeric_flex)

    # Booleans -> 0/1 float
    for b in BOOLEAN_COLS:
        if b in df.columns:
            s = df[b].astype(str).str.strip().str.lower()
            df[b] = s.isin(["true", "1", "yes", "y", "כן", "1.0", "1"]).astype(float)

    # Numeric
    use_num = [c for c in NUMERIC_COLS if c in df.columns]
    X_num = None
    if use_num:
        Xn = df[use_num].applymap(to_numeric_flex)
        Xn = Xn.fillna(pd.Series({c: Xn[c].median() for c in use_num}))
        scl = StandardScaler()
        Xn = scl.fit_transform(Xn.values)
        Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
        X_num = csr_matrix(Xn) * w_num

    # Categorical
    use_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    X_cat = None
    if use_cat:
        ohe = make_ohe()
        X_cat = ohe.fit_transform(df[use_cat].astype(str))
        row_norms = np.sqrt(X_cat.multiply(X_cat).sum(axis=1)) + 1e-12
        X_cat = X_cat.multiply(1.0 / row_norms) * w_cat

    # Boolean
    use_bool = [b for b in BOOLEAN_COLS if b in df.columns]
    X_bool = None
    if use_bool:
        Xb = df[use_bool].astype(float).values
        Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12)
        X_bool = csr_matrix(Xb) * w_bool

    # Embedding
    X_emb, emb_dim = build_embedding_block(df, EMBEDDING_COL)
    if X_emb is not None and w_emb != 1.0:
        X_emb = X_emb * w_emb

    # Stitch + record slices
    blocks = [m for m in [X_num, X_cat, X_bool, X_emb] if m is not None]
    if not blocks:
        raise RuntimeError("No feature blocks available. Check your column names.")

    slices = {}
    start = 0
    result = None
    for name, mat in [('num', X_num), ('cat', X_cat), ('bool', X_bool), ('emb', X_emb)]:
        if mat is None:
            continue
        ncols = mat.shape[1]
        end = start + ncols
        slices[name] = slice(start, end)
        result = mat if result is None else hstack([result, mat])
        start = end

    return df, result.tocsr(), slices

try:
    df, X_items, block_slices = build_blocks(df, (w_num, w_cat, w_bool, w_emb))
except Exception as e:
    st.error(f"Feature build failed: {e}")
    st.stop()

st.info(f"X_items shape: {X_items.shape} | blocks: {list(block_slices.keys())}")

# ======================= Similar items =======================
def similar_items(df, X_items, item_id, top_k=12, price_range_pct=None, same_loc=False, loc_key=None):
    idx_map = pd.Series(df.index, index=df["item_id"])
    if item_id not in idx_map:
        raise KeyError(f"item_id {item_id} not in dataset")
    i = int(idx_map[item_id])

    sims = cosine_similarity(X_items[i], X_items).ravel()
    order = np.argsort(-sims)
    cand = [j for j in order if j != i]

    # Price filter (vectorized on price_num)
    if price_range_pct and "price_num" in df.columns:
        p = df.loc[i, "price_num"]
        if pd.notna(p):
            lo, hi = p * (1 - price_range_pct), p * (1 + price_range_pct)
            mask = df["price_num"].between(lo, hi, inclusive="both").values
            cand = [j for j in cand if mask[j]]

    # Location filter
    if same_loc and loc_key in df.columns:
        v = str(df.loc[i, loc_key])
        loc_mask = (df[loc_key].astype(str).values == v)
        cand = [j for j in cand if loc_mask[j]]

    top = cand[:top_k]
    show_cols = [c for c in ["price", "price_num", "rooms", "size_sqm", loc_key,
                             "neighborhood", "apartment_style", "elevator", "ad_url"]
                 if c in df.columns]
    out = df.iloc[top][show_cols].copy()
    out.insert(0, "score", sims[top])
    return out.reset_index(drop=True)

# ======================= User profile + MMR =======================
def recommend_for_user(df, X_items, liked_item_ids, top_k=12, lambda_mmr=0.3, price_alpha=0.0):
    """Return diverse top-k using MMR; avoid np.matrix pitfalls."""
    if not liked_item_ids:
        return df.head(0)

    idx_map = pd.Series(df.index, index=df["item_id"])
    liked_idx = [int(idx_map[i]) for i in liked_item_ids if i in idx_map]
    if not liked_idx:
        return df.head(0)

    # centroid as 1xD ndarray (NOT np.matrix)
    if issparse(X_items):
        C = X_items[liked_idx].mean(axis=0)         # returns np.matrix
        C = np.asarray(C).reshape(1, -1)            # fix to ndarray
    else:
        C = X_items[liked_idx].mean(axis=0, keepdims=True)
        C = np.asarray(C)

    # L2 normalize centroid
    C = C / (np.linalg.norm(C) + 1e-12)

    # base similarity to all items
    sims = cosine_similarity(C, X_items).ravel()
    sims[liked_idx] = -1.0  # exclude seeds

    # candidate pool
    pool = np.argsort(-sims)[: max(200, top_k * 5)]
    if len(pool) == 0:
        return df.head(0)

    pool_vectors = X_items[pool]
    sim_pool = cosine_similarity(pool_vectors, pool_vectors)  # (<=200 x 200)

    # optional price penalty: farther from liked price median -> lower score
    base = sims[pool].astype(float)
    if "price_num" in df.columns and price_alpha > 0:
        liked_p = df.iloc[liked_idx]["price_num"].dropna()
        if not liked_p.empty:
            target_p = liked_p.median()
            p = df.iloc[pool]["price_num"].fillna(target_p).values
            penalty = -price_alpha * np.abs((p - target_p) / (target_p + 1e-9))
            base = base + penalty

    selected, selected_pos = [], []
    while len(selected) < min(top_k, len(pool)):
        best_cand, best_score = None, -1e9
        for pos, cand in enumerate(pool[:200]):
            if cand in selected:
                continue
            rel = base[pos]
            div = 0.0 if not selected_pos else float(sim_pool[pos, selected_pos].max())
            mmr = (1 - lambda_mmr) * rel - lambda_mmr * div
            if mmr > best_score:
                best_score, best_cand, best_pos = mmr, cand, pos
        selected.append(best_cand); selected_pos.append(best_pos)

    show_cols = [c for c in ["price","price_num","rooms","size_sqm","city","city_group",
                             "neighborhood","apartment_style","elevator","ad_url"]
                 if c in df.columns]
    out = df.iloc[selected][show_cols].copy()
    out.insert(0, "score", sims[selected])
    return out.reset_index(drop=True)

# ======================= Chat helpers =======================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

CITY_CANDIDATES = sorted({str(c).strip() for c in df.get('city', pd.Series([])).dropna().unique()})
CITY_GROUP_CANDIDATES = sorted({str(c).strip() for c in df.get('city_group', pd.Series([])).dropna().unique()})

BOOL_KEYWORDS = {
    'elevator': 'elevator',
    'accessible': 'wheelchair_access',
    'accessibility': 'wheelchair_access',
    'ac': 'air_conditioning',
    'air conditioning': 'air_conditioning',
    'bars': 'bars',
    'tornado ac': 'tornado_ac',
    'multi bolt': 'multi_bolt_doors',
}

def parse_user_message(msg: str) -> dict:
    """Parse a free-text prompt (English) into preference filters."""
    prefs = {}
    text = _norm(msg)

    # numbers (support "8k" -> 8000)
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text)]
    for m in re.findall(r"(\d+(?:\.\d+)?)\s*k", text):
        nums.append(float(m) * 1000)

    if nums:
        nums_sorted = sorted(nums)
        if any(w in text for w in ["between", "range", "to", "up to"]):
            prefs['price_min'] = nums_sorted[0]
            prefs['price_max'] = nums_sorted[-1]
        else:
            v = nums_sorted[-1]
            prefs['price_min'] = 0.8 * v
            prefs['price_max'] = 1.2 * v

    # rooms
    m_rooms = re.findall(r"(\d+(?:\.\d+)?)\s*room", text)
    if m_rooms:
        rs = sorted([float(r) for r in m_rooms])
        if 'to' in text and len(rs) >= 2:
            prefs['rooms_min'], prefs['rooms_max'] = rs[0], rs[-1]
        elif 'between' in text or 'range' in text:
            prefs['rooms_min'], prefs['rooms_max'] = rs[0], rs[-1]
        else:
            r = rs[-1]
            prefs['rooms_min'], prefs['rooms_max'] = r - 0.5, r + 0.5

    # city / city_group
    for city in CITY_CANDIDATES:
        if city and city.lower() in text:
            prefs['city'] = city
            break
    if 'city' not in prefs:
        for cg in CITY_GROUP_CANDIDATES:
            if cg and cg.lower() in text:
                prefs['city_group'] = cg
                break

    # booleans
    flags = {}
    for kw, col in BOOL_KEYWORDS.items():
        if kw in text:
            flags[col] = True
    if flags:
        prefs['bools'] = flags

    return prefs

def apply_prefs(df_in: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    dfc = df_in.copy()

    # price
    if 'price_num' not in dfc.columns and 'price' in dfc.columns:
        dfc['price_num'] = dfc['price'].apply(to_numeric_flex)
    if 'price_min' in prefs:
        dfc = dfc[dfc['price_num'] >= prefs['price_min']]
    if 'price_max' in prefs:
        dfc = dfc[dfc['price_num'] <= prefs['price_max']]

    # rooms
    if 'rooms_min' in prefs:
        dfc = dfc[dfc['rooms'] >= prefs['rooms_min']]
    if 'rooms_max' in prefs:
        dfc = dfc[dfc['rooms'] <= prefs['rooms_max']]

    # location
    if 'city' in prefs and 'city' in dfc.columns:
        dfc = dfc[dfc['city'].astype(str).str.lower() == prefs['city'].lower()]
    elif 'city_group' in prefs and 'city_group' in dfc.columns:
        dfc = dfc[dfc['city_group'].astype(str).str.lower() == prefs['city_group'].lower()]

    # booleans
    for col in prefs.get('bools', {}):
        if col in dfc.columns:
            s = dfc[col].astype(str).str.strip().str.lower()
            mask = s.isin(["true", "1", "yes", "y", "כן", "1.0", "1"])
            dfc = dfc[mask]

    return dfc

def rank_candidates_by_similarity(df_cand: pd.DataFrame, X_all, top_k=20):
    if len(df_cand) == 0:
        return df_cand
    idx = df_cand.index.to_numpy()
    Xi = X_all[idx]
    if Xi.shape[0] >= 2:
        C = Xi.mean(axis=0)               # np.matrix for sparse
        C = np.asarray(C).reshape(1, -1)  # ensure ndarray
        C = C / (np.linalg.norm(C) + 1e-12)
        sims = cosine_similarity(C, Xi).ravel()
        order = np.argsort(-sims)
        return df_cand.iloc[order].head(top_k)
    else:
        return df_cand.head(top_k)

# ======================= UI Tabs =======================
tab1, tab2, tab3 = st.tabs(["🔎 Similar to one", "👤 User profile (MMR)", "💬 Chatbot"])

with tab1:
    st.subheader("🔎 Similar to one listing")
    qid = st.selectbox("Choose item_id:", df["item_id"].tolist())
    col1, col2, col3 = st.columns(3)
    with col1:
        loc_key = st.selectbox("Location key", [c for c in ["city", "city_group"] if c in df.columns])
    with col2:
        same_loc = st.checkbox(f"Constrain to same {loc_key}", value=True)
    with col3:
        pr_pct = st.slider("Price range ±%", 0, 50, 20, step=5)

    if st.button("Find similar"):
        try:
            res = similar_items(
                df, X_items, qid,
                top_k=12,
                price_range_pct=(pr_pct/100 if pr_pct > 0 else None),
                same_loc=same_loc, loc_key=loc_key
            )
            if len(res) == 0:
                st.warning("No results — widen your ranges or disable filters.")
            if "ad_url" in res.columns:
                res["ad_url"] = res["ad_url"].apply(lambda x: f"<a href='{x}' target='_blank'>link</a>" if pd.notna(x) else "")
                st.write(res.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.dataframe(res, use_container_width=True)
        except Exception as e:
            st.error(f"Similar-items error: {e}")

with tab2:
    st.subheader("👤 User profile with diversity (MMR)")
    options = df["item_id"].tolist()
    liked = st.multiselect("Choose 1–5 liked listings:", options, max_selections=5)
    col1, col2 = st.columns(2)
    with col1:
        lambda_mmr = st.slider("Diversify (MMR λ)", 0.0, 1.0, 0.3, step=0.05)
    with col2:
        price_alpha = st.slider("Price adherence (α)", 0.0, 1.0, 0.0, step=0.05)

    if st.button("Recommend"):
        try:
            res = recommend_for_user(df, X_items, liked, top_k=12, lambda_mmr=lambda_mmr, price_alpha=price_alpha)
            if len(res) == 0:
                st.warning("No results — try different seeds or lower λ/α.")
            if "ad_url" in res.columns:
                res["ad_url"] = res["ad_url"].apply(lambda x: f"<a href='{x}' target='_blank'>link</a>" if pd.notna(x) else "")
                st.write(res.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.dataframe(res, use_container_width=True)
        except Exception as e:
            st.error(f"User-profile error: {e}")

with tab3:
    st.subheader("💬 Chat — describe what you want")
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_prefs" not in st.session_state:
        st.session_state.chat_prefs = {}

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🧹 Reset preferences"):
            st.session_state.chat_prefs = {}
            st.session_state.chat_history.append(("assistant", "Preferences reset. What are you looking for?"))
    with c2:
        if st.button("🗑️ Clear history"):
            st.session_state.chat_history = []

    # show history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    prompt = st.chat_input("E.g., 'Budget up to 7,500 in Tel Aviv, 3 rooms, elevator'")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.write(prompt)

        prefs = parse_user_message(prompt)
        st.session_state.chat_prefs.update(prefs)

        df_cand = apply_prefs(df, st.session_state.chat_prefs)
        n_found = len(df_cand)
        df_ranked = rank_candidates_by_similarity(df_cand, X_items, top_k=20)

        summary = [
            "Got your preferences:",
            f"- Price range: {int(st.session_state.chat_prefs.get('price_min', 0))}–{int(st.session_state.chat_prefs.get('price_max', 0))} ₪"
            if ('price_min' in st.session_state.chat_prefs or 'price_max' in st.session_state.chat_prefs) else "- Price range: (not specified)",
            f"- Rooms: {st.session_state.chat_prefs.get('rooms_min','?')}–{st.session_state.chat_prefs.get('rooms_max','?')}"
            if ('rooms_min' in st.session_state.chat_prefs or 'rooms_max' in st.session_state.chat_prefs) else "- Rooms: (not specified)",
        ]
        if 'city' in st.session_state.chat_prefs:
            summary.append(f"- City: {st.session_state.chat_prefs['city']}")
        if 'city_group' in st.session_state.chat_prefs:
            summary.append(f"- City group: {st.session_state.chat_prefs['city_group']}")
        if st.session_state.chat_prefs.get('bools'):
            summary.append("- Features: " + ", ".join([k for k,v in st.session_state.chat_prefs['bools'].items() if v]))

        reply = "\n".join(summary) + f"\n\nFound {n_found} matching listings (showing up to 20)."

        with st.chat_message("assistant"):
            st.write(reply)
            show_cols = [c for c in ["price","price_num","rooms","size_sqm","city","city_group",
                                     "neighborhood","apartment_style","elevator","ad_url"]
                         if c in df_ranked.columns]
            res = df_ranked[show_cols].copy().reset_index(drop=True)
            if "ad_url" in res.columns:
                res["ad_url"] = res["ad_url"].apply(lambda x: f"<a href='{x}' target='_blank'>link</a>" if pd.notna(x) else "")
                st.write(res.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.dataframe(res, use_container_width=True)

        st.session_state.chat_history.append(("assistant", reply))
