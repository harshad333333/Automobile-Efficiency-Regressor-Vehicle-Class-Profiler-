# 🏎️ Automobile Efficiency Regressor & Vehicle Class Profiler

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-00f2fe?style=for-the-badge&logo=github)](https://github.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![JavaScript ES6](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **An interactive Machine Learning data science platform & simulation web application for automobile fuel efficiency optimization and unsupervised physical design archetype profiling.**

---

## 📌 Problem Statement & Overview

Car manufacturers face a multi-dimensional engineering challenge: balancing powertrain output (horsepower, displacement, cylinders) against structural curb weight to maximize fuel efficiency (Miles Per Gallon - MPG) while minimizing carbon emissions.

This project delivers a dual-paradigm Machine Learning solution on the canonical **UCI Auto MPG Dataset** (398 production vehicles):
1. **Vehicle Class Profiler (Unsupervised Learning)**: Groups vehicles into 4 distinct physical design archetypes using standardized K-Means clustering and geometric radar analysis.
2. **Automobile Efficiency Regressor (Supervised Regularized Learning)**: Solves multicollinearity across engine displacement and vehicle mass using **Ridge ($L_2$)**, **Lasso ($L_1$)**, and **ElasticNet** regression models with real-time client-side hyperparameter $\alpha$ tuning.

---

## 🌟 Key Features

- ⚡ **Zero-Latency In-Browser ML Engine**: Analytical Ridge solver and iterative Coordinate Descent Lasso/ElasticNet solver written directly in modern JavaScript.
- 🎛️ **Interactive What-If Configurator**: Dynamic sliders for Curb Weight, Horsepower, Cylinders, Displacement, 0-60 Acceleration, Year, and Origin.
- ⏱️ **Speedometer-Style MPG Gauge**: Real-time animated needle physics, 95% confidence intervals, and efficiency grades ($A+$ to $F$).
- 🌿 **Environmental & Economic Intelligence**: Live estimates for annual fuel cost (@ 15,000 miles/yr @ $3.60/gal), savings vs. 20 MPG fleet averages, and annual metric tons of $\text{CO}_2$ output.
- 🎯 **Reverse Target MPG Optimizer**: Calculates recommended weight and horsepower reductions to meet specific fuel economy targets.
- 📊 **Multidimensional Visualizations**:
  - Radar Specification Chart for the 4 physical archetypes.
  - 2D/3D-like Scatter Projection with customizable axes (Weight vs. MPG, HP vs. MPG).
  - Regularization Shrinkage Paths ($\beta_j(\alpha)$ curves on log-scale).
  - SHAP-style Feature Impact waterfall bars.
  - Actual vs. Predicted Parity plot with $45^\circ$ perfect-fit line.
- 🔍 **Searchable Auto MPG Dataset Explorer**: Filter 398 historical vehicles by origin and archetype with instantaneous "Load into Simulator" capability.
- 🚀 **100% Static & GitHub Pages Ready**: Deployable with zero server maintenance.

---

## 📐 Mathematical Formulation

### 1. Ridge Regression ($L_2$ Regularization)
Minimizes residual sum of squares subject to an $L_2$ coefficient penalty to handle high collinearity between weight and displacement:
$$\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2 \implies \mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

### 2. Lasso Regression ($L_1$ Regularization)
Performs sparse feature selection by driving redundant feature weights to exact zero via Cyclical Coordinate Descent with Soft-Thresholding:
$$\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$$
$$w_j \leftarrow \frac{S\left( \sum_{i=1}^n x_{ij}(y_i - \hat{y}_i^{(-j)}), \alpha \right)}{\sum_{i=1}^n x_{ij}^2}, \quad \text{where } S(z, \gamma) = \text{sign}(z)\max(0, |z| - \gamma)$$

### 3. ElasticNet Regularization
Combines $L_1$ sparsity and $L_2$ grouped shrinkage:
$$\min_{\mathbf{w}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{w}\|_2^2$$

### 4. K-Means Archetype Profiling
Partitions vehicles into $k=4$ clusters minimizing within-cluster sum-of-squares in normalized 6D space:
$$\arg\min_{\mathbf{S}} \sum_{i=1}^k \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \boldsymbol{\mu}_i\|^2$$

| Archetype Cluster | Avg MPG | Avg Horsepower | Avg Curb Weight | Key Traits |
| :--- | :---: | :---: | :---: | :--- |
| **Eco Commuter** | 31.8 MPG | 77.3 HP | 2,347 lbs | Lightweight, agile city cars (Corolla, Civic, Rabbit) |
| **Balanced Daily Sedan** | 25.2 MPG | 81.4 HP | 2,302 lbs | Well-rounded commuter sedans |
| **Mid-Size Cruiser** | 19.9 MPG | 103.3 HP | 3,280 lbs | 6-cylinder highway cruisers |
| **V8 Muscle & Heavy Hauler** | 14.4 MPG | 162.4 HP | 4,158 lbs | High torque V8 muscle cars (Mustang, Chevelle, Firebird) |

---

## 🏆 Model Benchmark & Performance

| Model | Test $R^2$ | RMSE | MAE | 5-Fold Cross-Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge ($\alpha=1.0$)** | **0.8474** | **2.86 MPG** | **2.25 MPG** | 0.595 |
| **Lasso ($\alpha=0.5$)** | **0.8487** | **2.85 MPG** | **2.21 MPG** | 0.568 |
| **ElasticNet ($\alpha=0.1, \rho=0.5$)** | **0.8472** | **2.86 MPG** | **2.26 MPG** | 0.580 |
| **Standard OLS** | **0.8476** | **2.86 MPG** | **2.25 MPG** | 0.596 |

---

## 📁 Repository Structure

```
automobile/
├── .github/
│   └── workflows/
│       └── deploy.yml            # Automated GitHub Pages CI/CD workflow
├── css/
│   └── styles.css               # Dark luxury cyber-automotive styling & animations
├── js/
│   ├── data.js                  # Cleaned 398-vehicle database & scaling constants
│   ├── ml-engine.js             # Client-side Ridge, Lasso, ElasticNet & K-Means solver
│   ├── charts.js                # Chart.js visualizers (Radar, Scatter, Shrinkage, Waterfall)
│   └── app.js                   # Reactive UI controller & What-If event bus
├── data/
│   └── auto_mpg.csv             # Cleaned Auto MPG dataset (CSV)
├── python/
│   ├── train_and_evaluate.py    # Python standalone ML pipeline (Scikit-Learn)
│   └── requirements.txt         # Python dependencies
├── index.html                   # Main interactive web application
├── README.md                    # Project documentation & GitHub guide
└── .gitignore                   # Git ignore patterns
```

---

## 🚀 How to Deploy on GitHub in 2 Minutes

### Method 1: Git CLI (Recommended)
1. Initialize git and commit the files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Automobile Efficiency Regressor & Profiler"
   ```
2. Create a new public repository on [GitHub](https://github.com/new) named `automobile-efficiency-profiler`.
3. Link and push to GitHub:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/automobile-efficiency-profiler.git
   git push -u origin main
   ```
4. **Enable GitHub Pages**:
   - Navigate to your repository on GitHub &rarr; **Settings** &rarr; **Pages**.
   - Under **Build and deployment > Source**, select **Deploy from a branch**.
   - Choose Branch: `main` and Folder: `/ (root)` &rarr; Click **Save**.
   - Your live site will be ready at: `https://YOUR_USERNAME.github.io/automobile-efficiency-profiler/`

---

## 💻 Running the Python ML Pipeline Locally

If you wish to explore the data science scripts or re-train the models locally:

```bash
# 1. Install dependencies
pip install -r python/requirements.txt

# 2. Run training, clustering & cross-validation
python python/train_and_evaluate.py
```

---

## 📜 Dataset Citation

- **Dataset**: Auto MPG Dataset
- **Origin**: StatLib Library, Carnegie Mellon University (1993)
- **Repository**: [UCI Machine Learning Repository: Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
- **Instances**: 398 vehicles (continuous and discrete attributes)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
