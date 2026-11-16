# Israeli Real Estate Analysis & Apartment Recommendations

An academic data-science project connecting listing collection, exploratory analysis, clustering, model interpretation, and an interactive apartment recommender.

## Engineering highlights
- A scraping pipeline with parsing, browser fetching, database integration, and embedding-related components.
- Exploratory analysis and feature preprocessing for property attributes.
- Clustering experiments, cluster profiles, and optimization results.
- Model interpretation notebooks and visualizations.
- A Streamlit recommendation app combining weighted numerical, categorical, and boolean features with cosine similarity; description embeddings are optional.

## Explore the work
| Area | Entry point |
| --- | --- |
| Data collection | `Yad2_Scraper/` |
| Exploration | `Preprocessing_Exploratory/EDA.ipynb` |
| Clustering | `Clustering/Stage2_Clustering.ipynb` |
| Interpretation | `Model_Explainability/Stage_4.ipynb` |
| Optimization | `Model_Optimization_Clustering/stage5_realestate_clustering.ipynb` |
| Interactive app | `Recommendetion_system/app.py` |

The application directory retains its original spelling.

## Local setup
From a Python virtual environment:
```bash
pip install -r requirements.txt
streamlit run Recommendetion_system/app.py
```
Review the app's CSV-loading logic and configure an authorized dataset before running it. Setup commands are provided as a starting point; a clean-environment rerun has not yet been documented.

## Data and responsible use
Listing data may include addresses and links to third-party advertisements. Use only data you are authorized to process, respect source terms, and avoid republishing personal information. The repository should not be interpreted as granting rights to third-party data. Do not expose database credentials or enable scraping as part of a public demo.

## Project status
This is a research and learning project, not a production housing or valuation service. Notebook outputs document experiments, but a unified benchmark and reproducible end-to-end evaluation remain future improvements.
