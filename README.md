# Yad2 Real Estate: Scraping, Clustering & Recommendations

An end-to-end data project on Israeli rental listings from Yad2. It scrapes the listings, cleans and explores them, clusters similar apartments, explains the model, and serves a recommendation app.

**[Live demo (Streamlit)](https://yad2-real-estate-project-duvfcfnpm5btrjvpaw3srp.streamlit.app/)**

## Pipeline

| Stage | Folder | What it does |
|---|---|---|
| 1. Scraping | `Yad2_Scraper/` | Modular scraper (Selenium + BeautifulSoup) with retries and logging. Listings are stored in MongoDB, and optional text embeddings come from a local embedding model. |
| 2. Preprocessing & EDA | `Preprocessing_Exploratory/` | Cleaning, type fixing and exploration: price per m², neighbourhood demand, student-relevant features. |
| 3. Clustering | `Clustering/` | K-Means, agglomerative (ward / average / complete) and DBSCAN, compared on silhouette, Calinski-Harabasz and Davies-Bouldin. |
| 4. Explainability | `Model_Explainability/` | SHAP summary and force plots, plus a depth-3 surrogate decision tree with a fidelity check, to explain what drives each cluster. |
| 5. Optimisation | `Model_Optimization_Clustering/` | Hyper-parameter search across 68 clustering configurations, ranked by a weighted score of the three metrics. |
| 6. Recommender | `Recommendetion_system/app.py` | Streamlit app. Listings are encoded as weighted feature blocks (numeric, one-hot and embeddings), and similar apartments are found by cosine similarity, with price-range and location filters. |

## Running

```bash
pip install -r requirements.txt
streamlit run Recommendetion_system/app.py
```

A sample of the data is in `Data/sample_realestate.csv`. The scraper reads its settings (MongoDB URI, batch size, concurrency) from environment variables. See `Yad2_Scraper/README.md`.

## Tech

Python · Selenium · BeautifulSoup · MongoDB · pandas · scikit-learn · SHAP · Streamlit
