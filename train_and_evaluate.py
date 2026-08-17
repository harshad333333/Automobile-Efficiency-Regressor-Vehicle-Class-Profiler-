"""
Automobile Efficiency Regressor & Vehicle Class Profiler
Standalone Python Data Science Pipeline (scikit-learn, pandas, numpy)

Features:
- Auto MPG Dataset ingestion & preprocessing (missing value handling, scaling)
- Unsupervised Learning: K-Means Clustering (k=4) for Vehicle Physical Archetyping
- Supervised Learning: OLS, Ridge (L2), Lasso (L1), and ElasticNet Regularized Regression
- Evaluation: R2, RMSE, MAE, Cross-Validation, Regularization Paths
- Export: Prepares clean dataset JSON and coefficients for Client-side Web Dashboard
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_and_clean_data():
    print("Loading Auto MPG dataset...")
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "auto_mpg.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    import io
    urls = [
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"
    ]
    
    df = None
    for url in urls:
        try:
            print(f"Fetching from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read()
                df = pd.read_csv(io.BytesIO(content))
            if df is not None and len(df) > 0:
                print(f"Successfully loaded {len(df)} records from {url}")
                break
        except Exception as e:
            print(f"Fetch failed for {url}: {e}")
            continue

    # Standardize column names
    rename_dict = {}
    for c in df.columns:
        clean_c = c.lower().strip().replace(' ', '_')
        if clean_c in ['year', 'modelyear']:
            rename_dict[c] = 'model_year'
        elif clean_c in ['name', 'carname']:
            rename_dict[c] = 'car_name'
        else:
            rename_dict[c] = clean_c
    df.rename(columns=rename_dict, inplace=True)
    
    # Standardize origin to numeric 1: USA, 2: Europe, 3: Japan if strings
    origin_map = {'usa': 1, 'europe': 2, 'japan': 3, '1': 1, '2': 2, '3': 3, 1: 1, 2: 2, 3: 3}
    df['origin'] = df['origin'].astype(str).str.lower().map(origin_map).fillna(1).astype(int)
    
    if 'horsepower' in df.columns and df['horsepower'].dtype == object:
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

    # Impute missing horsepower values with median based on cylinders group
    if df['horsepower'].isnull().sum() > 0:
        df['horsepower'] = df.groupby('cylinders')['horsepower'].transform(
            lambda x: x.fillna(x.median())
        )
        # Global median fallback if any still null
        df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    
    # Clean car names
    df['car_name'] = df['car_name'].astype(str).str.strip().str.replace('"', '')
    
    # Save clean CSV
    df.to_csv(raw_path, index=False)
    print(f"Cleaned dataset saved to {raw_path}")
    return df

def run_pipeline():
    df = load_and_clean_data()
    
    feature_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
    physical_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    
    X = df[feature_cols].copy()
    y = df['mpg'].values
    
    # One-hot encode origin (1: USA, 2: Europe, 3: Japan)
    # For standard linear models, we can keep continuous + one-hot or standard scaled
    # Let's scale features for regularized regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Unsupervised Clustering (Vehicle Class Profiler)
    print("\n--- Clustering & Vehicle Profiling (K-Means k=4) ---")
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(df[physical_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    # Sort cluster names based on average MPG/Weight for intuitive archetypes
    cluster_stats = df.groupby('cluster').agg({
        'mpg': 'mean',
        'weight': 'mean',
        'horsepower': 'mean',
        'displacement': 'mean',
        'cylinders': 'mean',
        'acceleration': 'mean'
    }).reset_index()
    
    # Assign semantic archetype labels
    # Sort clusters by average MPG descending:
    # Top MPG -> Eco Commuter
    # 2nd -> Balanced Daily Sedan
    # 3rd -> High-Displacement Cruiser
    # Lowest MPG -> V8 Muscle / Heavy Hauler
    cluster_stats = cluster_stats.sort_values('mpg', ascending=False).reset_index(drop=True)
    archetype_names = [
        "Eco Commuter",
        "Balanced Daily Sedan",
        "Mid-Size Cruiser",
        "V8 Muscle & Heavy Hauler"
    ]
    cluster_mapping = {row['cluster']: archetype_names[i] for i, row in cluster_stats.iterrows()}
    df['archetype'] = df['cluster'].map(cluster_mapping)
    
    print("Cluster Archetype Summary:")
    for i, name in enumerate(archetype_names):
        c_id = cluster_stats.loc[i, 'cluster']
        c_df = df[df['cluster'] == c_id]
        print(f"[{i}] {name}: N={len(c_df)}, Avg MPG={c_df['mpg'].mean():.1f}, Avg HP={c_df['horsepower'].mean():.1f}, Avg Wt={c_df['weight'].mean():.0f} lbs")

    # 2. Supervised Learning (Efficiency Regressors)
    print("\n--- Training Regularized Regression Models ---")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'OLS': LinearRegression(),
        'Ridge (L2, alpha=1.0)': Ridge(alpha=1.0, random_state=42),
        'Ridge (L2, alpha=10.0)': Ridge(alpha=10.0, random_state=42),
        'Lasso (L1, alpha=0.1)': Lasso(alpha=0.1, random_state=42),
        'Lasso (L1, alpha=0.5)': Lasso(alpha=0.5, random_state=42),
        'ElasticNet (alpha=0.1, l1=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'ElasticNet (alpha=0.5, l1=0.7)': ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        results[name] = {
            'r2': round(float(r2), 4),
            'rmse': round(float(rmse), 4),
            'mae': round(float(mae), 4),
            'cv_r2_mean': round(float(np.mean(cv_scores)), 4),
            'cv_r2_std': round(float(np.std(cv_scores)), 4),
            'intercept': round(float(model.intercept_), 4),
            'coefficients': {col: round(float(coef), 4) for col, coef in zip(feature_cols, model.coef_)}
        }
        print(f"{name:32} | R²: {r2:.4f} | RMSE: {rmse:.3f} MPG | MAE: {mae:.3f} MPG | 5-Fold CV R²: {np.mean(cv_scores):.4f}")

    # 3. Regularization Path for alpha in [1e-3 to 1e3]
    alphas = np.logspace(-3, 2.5, 40).tolist()
    ridge_paths = {col: [] for col in feature_cols}
    lasso_paths = {col: [] for col in feature_cols}
    
    for a in alphas:
        r_mod = Ridge(alpha=a).fit(X_scaled, y)
        l_mod = Lasso(alpha=a, max_iter=2000).fit(X_scaled, y)
        for i, col in enumerate(feature_cols):
            ridge_paths[col].append(round(float(r_mod.coef_[i]), 4))
            lasso_paths[col].append(round(float(l_mod.coef_[i]), 4))

    # 4. Export Web App Data Asset (js/data.js)
    js_dir = os.path.join(os.path.dirname(__file__), "..", "js")
    os.makedirs(js_dir, exist_ok=True)
    js_file = os.path.join(js_dir, "data.js")
    
    # Structure full dataset for client-side
    records = []
    for _, row in df.iterrows():
        records.append({
            'name': row['car_name'],
            'mpg': float(row['mpg']),
            'cylinders': int(row['cylinders']),
            'displacement': float(row['displacement']),
            'horsepower': float(row['horsepower']),
            'weight': float(row['weight']),
            'acceleration': float(row['acceleration']),
            'model_year': int(row['model_year']),
            'origin': int(row['origin']),
            'cluster': int(row['cluster']),
            'archetype': str(row['archetype'])
        })
        
    centroids = []
    for c_id, center in enumerate(kmeans.cluster_centers_):
        # Convert back to unscaled center values for easy interpretation
        unscaled = center * cluster_scaler.scale_ + cluster_scaler.mean_
        centroids.append({
            'cluster_id': int(c_id),
            'archetype': cluster_mapping[c_id],
            'scaled_center': [round(float(v), 4) for v in center],
            'center': {col: round(float(unscaled[i]), 2) for i, col in enumerate(physical_features)}
        })
        
    scaler_params = {
        'features': feature_cols,
        'mean': [round(float(m), 4) for m in scaler.mean_],
        'scale': [round(float(s), 4) for s in scaler.scale_]
    }
    
    cluster_scaler_params = {
        'features': physical_features,
        'mean': [round(float(m), 4) for m in cluster_scaler.mean_],
        'scale': [round(float(s), 4) for s in cluster_scaler.scale_]
    }

    data_payload = {
        'records': records,
        'feature_cols': feature_cols,
        'physical_features': physical_features,
        'scaler': scaler_params,
        'cluster_scaler': cluster_scaler_params,
        'centroids': centroids,
        'archetypes': archetype_names,
        'model_benchmarks': results,
        'regularization_paths': {
            'alphas': [round(float(a), 4) for a in alphas],
            'ridge': ridge_paths,
            'lasso': lasso_paths
        },
        'summary_stats': {
            'total_vehicles': len(df),
            'avg_mpg': round(float(df['mpg'].mean()), 2),
            'min_mpg': round(float(df['mpg'].min()), 2),
            'max_mpg': round(float(df['mpg'].max()), 2),
            'avg_hp': round(float(df['horsepower'].mean()), 2),
            'avg_weight': round(float(df['weight'].mean()), 2)
        }
    }
    
    with open(js_file, 'w', encoding='utf-8') as f:
        f.write("// Auto MPG Dataset & Precomputed ML Model Parameters for In-Browser Engine\n")
        f.write("window.AUTO_DATA = " + json.dumps(data_payload, indent=2) + ";\n")
        
    print(f"\nSuccessfully generated {js_file} with {len(records)} vehicle records and ML parameters.")

if __name__ == '__main__':
    run_pipeline()
