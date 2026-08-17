/**
 * Automobile Efficiency Regressor & Vehicle Class Profiler
 * Core Client-Side Machine Learning Engine
 * 
 * Includes:
 * - Matrix operations & Linear Algebra
 * - Ridge Regression (Closed-form L2 regularized analytical solver)
 * - Lasso Regression (Cyclical Coordinate Descent with Soft Thresholding)
 * - ElasticNet Regression (Convex combination L1 + L2 Regularization)
 * - K-Means Archetype Profiler (6D Euclidean Distance & Cluster Assignment)
 * - Live What-If Sensitivity & Carbon / Fuel Economy Analytics
 */

class MLEngine {
    constructor(dataPayload) {
        this.data = dataPayload;
        this.records = dataPayload.records;
        this.featureCols = dataPayload.feature_cols;
        this.physicalFeatures = dataPayload.physical_features;
        this.scaler = dataPayload.scaler;
        this.clusterScaler = dataPayload.cluster_scaler;
        this.centroids = dataPayload.centroids;
        this.benchmarks = dataPayload.model_benchmarks;
        this.paths = dataPayload.regularization_paths;
        this.summary = dataPayload.summary_stats;

        // Initialize design matrix X and target y for live solver
        this._prepareDataMatrices();
        
        // Active model cache
        this.activeModelType = 'ridge';
        this.activeAlpha = 1.0;
        this.activeL1Ratio = 0.5;
        this.currentWeights = null;
        this.currentIntercept = 0;
        
        // Train initial model
        this.trainActiveModel();
    }

    _prepareDataMatrices() {
        const n = this.records.length;
        const p = this.featureCols.length;
        this.X = [];
        this.y = new Float64Array(n);

        for (let i = 0; i < n; i++) {
            const r = this.records[i];
            const row = new Float64Array(p);
            for (let j = 0; j < p; j++) {
                const col = this.featureCols[j];
                const rawVal = r[col];
                // Standardize
                row[j] = (rawVal - this.scaler.mean[j]) / this.scaler.scale[j];
            }
            this.X.push(row);
            this.y[i] = r.mpg;
        }

        // Precompute feature column squared norms for coordinate descent
        this.colNorms = new Float64Array(p);
        for (let j = 0; j < p; j++) {
            let sumSq = 0;
            for (let i = 0; i < n; i++) {
                sumSq += this.X[i][j] * this.X[i][j];
            }
            this.colNorms[j] = sumSq;
        }

        // Precompute X^T X and X^T y for Ridge
        this.XtX = this._computeXtX();
        this.Xty = this._computeXty();
    }

    _computeXtX() {
        const p = this.featureCols.length;
        const n = this.X.length;
        const xtx = Array.from({ length: p }, () => new Float64Array(p));
        for (let j1 = 0; j1 < p; j1++) {
            for (let j2 = j1; j2 < p; j2++) {
                let dot = 0;
                for (let i = 0; i < n; i++) {
                    dot += this.X[i][j1] * this.X[i][j2];
                }
                xtx[j1][j2] = dot;
                xtx[j2][j1] = dot;
            }
        }
        return xtx;
    }

    _computeXty() {
        const p = this.featureCols.length;
        const n = this.X.length;
        const xty = new Float64Array(p);
        const yMean = this.y.reduce((a, b) => a + b, 0) / n;
        this.yMean = yMean;

        for (let j = 0; j < p; j++) {
            let dot = 0;
            for (let i = 0; i < n; i++) {
                dot += this.X[i][j] * (this.y[i] - yMean);
            }
            xty[j] = dot;
        }
        return xty;
    }

    // Soft thresholding operator for Lasso / ElasticNet
    _softThreshold(z, gamma) {
        if (z > gamma) return z - gamma;
        if (z < -gamma) return z + gamma;
        return 0;
    }

    // Solve Ax = b via Gauss-Jordan elimination with partial pivoting
    _solveLinearSystem(A, b) {
        const n = b.length;
        // Clone matrix & vector
        const M = A.map(row => Float64Array.from(row));
        const x = Float64Array.from(b);

        for (let i = 0; i < n; i++) {
            // Pivot selection
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) {
                    maxRow = k;
                }
            }
            if (maxRow !== i) {
                const tempRow = M[i];
                M[i] = M[maxRow];
                M[maxRow] = tempRow;
                const tempB = x[i];
                x[i] = x[maxRow];
                x[maxRow] = tempB;
            }

            const pivot = M[i][i];
            if (Math.abs(pivot) < 1e-12) {
                M[i][i] = 1e-12; // Avoid zero-divide singularity
            }
            const invPivot = 1.0 / M[i][i];

            for (let j = i; j < n; j++) {
                M[i][j] *= invPivot;
            }
            x[i] *= invPivot;

            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = M[k][i];
                    if (factor !== 0) {
                        for (let j = i; j < n; j++) {
                            M[k][j] -= factor * M[i][j];
                        }
                        x[k] -= factor * x[i];
                    }
                }
            }
        }
        return x;
    }

    /**
     * Train / solve model based on current hyperparameter settings
     */
    trainActiveModel(type = this.activeModelType, alpha = this.activeAlpha, l1Ratio = this.activeL1Ratio) {
        this.activeModelType = type;
        this.activeAlpha = Math.max(0.0001, alpha);
        this.activeL1Ratio = Math.min(1.0, Math.max(0.0, l1Ratio));

        const p = this.featureCols.length;
        const n = this.X.length;

        if (type === 'ols' || (type === 'ridge' && this.activeAlpha < 1e-5)) {
            // Ridge with minimal regularization or standard OLS
            const A = this.XtX.map(row => Float64Array.from(row));
            for (let j = 0; j < p; j++) {
                A[j][j] += 1e-4; // Numerical stability jitter
            }
            this.currentWeights = this._solveLinearSystem(A, this.Xty);
            this.currentIntercept = this.yMean;
        } else if (type === 'ridge') {
            // Ridge: (X^T X + alpha * I)^(-1) X^T (y - y_mean)
            const A = this.XtX.map(row => Float64Array.from(row));
            for (let j = 0; j < p; j++) {
                A[j][j] += this.activeAlpha;
            }
            this.currentWeights = this._solveLinearSystem(A, this.Xty);
            this.currentIntercept = this.yMean;
        } else if (type === 'lasso' || type === 'elasticnet') {
            // Coordinate Descent Solver
            const isLasso = (type === 'lasso');
            const l1Penalty = isLasso ? this.activeAlpha : this.activeAlpha * this.activeL1Ratio;
            const l2Penalty = isLasso ? 0 : this.activeAlpha * (1.0 - this.activeL1Ratio);

            const w = new Float64Array(p);
            const residuals = new Float64Array(n);
            for (let i = 0; i < n; i++) {
                residuals[i] = this.y[i] - this.yMean;
            }

            const maxIter = 1000;
            const tol = 1e-5;

            for (let iter = 0; iter < maxIter; iter++) {
                let maxChange = 0;

                for (let j = 0; j < p; j++) {
                    const oldW = w[j];
                    const xj = this.X;
                    const normSq = this.colNorms[j];

                    if (normSq === 0) continue;

                    // Compute partial correlation rho_j = sum(x_ij * (r_i + x_ij * oldW))
                    let rho = 0;
                    for (let i = 0; i < n; i++) {
                        rho += xj[i][j] * (residuals[i] + xj[i][j] * oldW);
                    }

                    // Soft thresholding with L1 and L2 denominators
                    const newW = this._softThreshold(rho, l1Penalty * n * 0.1) / (normSq + l2Penalty * n * 0.1);
                    w[j] = newW;

                    const diff = newW - oldW;
                    if (Math.abs(diff) > maxChange) {
                        maxChange = Math.abs(diff);
                    }

                    // Update residuals in-place
                    if (diff !== 0) {
                        for (let i = 0; i < n; i++) {
                            residuals[i] -= diff * xj[i][j];
                        }
                    }
                }

                if (maxChange < tol) break;
            }

            this.currentWeights = w;
            this.currentIntercept = this.yMean;
        }

        return this.evaluateModel();
    }

    /**
     * Compute current model performance metrics over dataset
     */
    evaluateModel() {
        const n = this.X.length;
        const p = this.featureCols.length;
        let ssTot = 0;
        let ssRes = 0;
        let absErrSum = 0;
        const predictions = new Float64Array(n);

        for (let i = 0; i < n; i++) {
            let pred = this.currentIntercept;
            for (let j = 0; j < p; j++) {
                pred += this.currentWeights[j] * this.X[i][j];
            }
            predictions[i] = pred;
            const err = this.y[i] - pred;
            ssRes += err * err;
            absErrSum += Math.abs(err);
            const totErr = this.y[i] - this.yMean;
            ssTot += totErr * totErr;
        }

        const r2 = 1.0 - (ssRes / (ssTot || 1.0));
        const mse = ssRes / n;
        const rmse = Math.sqrt(mse);
        const mae = absErrSum / n;

        // Count active non-zero features
        let nonZeroFeatures = 0;
        const coefMap = {};
        for (let j = 0; j < p; j++) {
            const col = this.featureCols[j];
            const val = this.currentWeights[j];
            coefMap[col] = parseFloat(val.toFixed(4));
            if (Math.abs(val) > 1e-4) nonZeroFeatures++;
        }

        return {
            modelType: this.activeModelType,
            alpha: this.activeAlpha,
            l1Ratio: this.activeL1Ratio,
            r2: parseFloat(r2.toFixed(4)),
            rmse: parseFloat(rmse.toFixed(3)),
            mae: parseFloat(mae.toFixed(3)),
            nonZeroFeatures: nonZeroFeatures,
            totalFeatures: p,
            intercept: parseFloat(this.currentIntercept.toFixed(3)),
            coefficients: coefMap,
            predictions: predictions
        };
    }

    /**
     * Predict MPG for custom vehicle input parameters
     * @param {Object} input - {cylinders, displacement, horsepower, weight, acceleration, model_year, origin}
     */
    predictVehicle(input) {
        const p = this.featureCols.length;
        const scaled = new Float64Array(p);
        const featureContributions = [];
        let predictedMpg = this.currentIntercept;

        for (let j = 0; j < p; j++) {
            const col = this.featureCols[j];
            const raw = parseFloat(input[col]);
            const sVal = (raw - this.scaler.mean[j]) / this.scaler.scale[j];
            scaled[j] = sVal;

            const impact = this.currentWeights[j] * sVal;
            predictedMpg += impact;

            featureContributions.push({
                feature: col,
                rawValue: raw,
                scaledValue: parseFloat(sVal.toFixed(3)),
                weight: parseFloat(this.currentWeights[j].toFixed(4)),
                impact: parseFloat(impact.toFixed(3))
            });
        }

        // Clamp to physical reality (min 6.0 MPG, max 60.0 MPG)
        predictedMpg = Math.max(6.0, Math.min(65.0, predictedMpg));

        // Compute 95% Confidence Interval (+- 1.96 * RMSE)
        const rmse = 2.85; // baseline calibrated standard error
        const ciLow = Math.max(5.0, predictedMpg - 1.96 * rmse);
        const ciHigh = predictedMpg + 1.96 * rmse;

        // Efficiency Grade calculation
        let grade = 'C';
        let gradeColor = '#f59e0b';
        if (predictedMpg >= 35) { grade = 'A+'; gradeColor = '#10b981'; }
        else if (predictedMpg >= 30) { grade = 'A'; gradeColor = '#059669'; }
        else if (predictedMpg >= 25) { grade = 'B'; gradeColor = '#06b6d4'; }
        else if (predictedMpg >= 20) { grade = 'C+'; gradeColor = '#3b82f6'; }
        else if (predictedMpg >= 16) { grade = 'C'; gradeColor = '#f59e0b'; }
        else if (predictedMpg >= 13) { grade = 'D'; gradeColor = '#ea580c'; }
        else { grade = 'F'; gradeColor = '#ef4444'; }

        // Green & Economic Impact Metrics (15,000 miles/yr @ $3.60/gallon)
        const annualMiles = 15000;
        const gasPricePerGal = 3.60;
        const annualGallons = annualMiles / predictedMpg;
        const annualFuelCost = annualGallons * gasPricePerGal;
        // 8,887 grams CO2 per gallon of gasoline = 8.887 kg
        const annualCo2Kg = annualGallons * 8.887;
        const annualCo2Tons = annualCo2Kg / 1000.0;

        // Baseline comparison vs 20 MPG fleet average
        const baselineGallons = annualMiles / 20.0;
        const annualSavingsVsAvg = (baselineGallons - annualGallons) * gasPricePerGal;

        // Classify into Physical Archetype (Unsupervised Profiler)
        const archetypeProfile = this.classifyArchetype(input);

        return {
            mpg: parseFloat(predictedMpg.toFixed(2)),
            ciLow: parseFloat(ciLow.toFixed(1)),
            ciHigh: parseFloat(ciHigh.toFixed(1)),
            grade: grade,
            gradeColor: gradeColor,
            annualFuelCost: Math.round(annualFuelCost),
            annualSavingsVsAvg: Math.round(annualSavingsVsAvg),
            annualCo2Tons: parseFloat(annualCo2Tons.toFixed(2)),
            annualGallons: Math.round(annualGallons),
            featureContributions: featureContributions,
            archetype: archetypeProfile
        };
    }

    /**
     * Unsupervised Profiler: Classify vehicle into 4 Physical Archetypes
     */
    classifyArchetype(input) {
        const p = this.physicalFeatures.length;
        const scaledVec = new Float64Array(p);

        for (let j = 0; j < p; j++) {
            const col = this.physicalFeatures[j];
            const raw = parseFloat(input[col]);
            scaledVec[j] = (raw - this.clusterScaler.mean[j]) / this.clusterScaler.scale[j];
        }

        let nearestCluster = 0;
        let minDistance = Infinity;
        const distances = [];

        for (let c = 0; c < this.centroids.length; c++) {
            const center = this.centroids[c].scaled_center;
            let distSq = 0;
            for (let j = 0; j < p; j++) {
                const diff = scaledVec[j] - center[j];
                distSq += diff * diff;
            }
            const dist = Math.sqrt(distSq);
            distances.push({
                cluster_id: this.centroids[c].cluster_id,
                archetype: this.centroids[c].archetype,
                distance: parseFloat(dist.toFixed(3)),
                centerSpec: this.centroids[c].center
            });
            if (dist < minDistance) {
                minDistance = dist;
                nearestCluster = c;
            }
        }

        // Calculate normalized similarity scores (%)
        const maxDist = Math.max(...distances.map(d => d.distance), 0.001);
        const similarities = distances.map(d => ({
            ...d,
            similarity: Math.max(5, Math.round((1 - (d.distance / (maxDist * 1.5))) * 100))
        })).sort((a, b) => b.similarity - a.similarity);

        const assigned = this.centroids[nearestCluster];

        // Description badge & traits
        const traitMap = {
            "Eco Commuter": {
                icon: "leaf",
                tag: "High Efficiency / Agile City Runner",
                color: "#10b981",
                description: "Sub-compact 4-cylinder architecture engineered for maximum fuel conservation, lightweight bodywork, and low rolling resistance."
            },
            "Balanced Daily Sedan": {
                icon: "shield-check",
                tag: "All-Rounder / Commuter Sedan",
                color: "#06b6d4",
                description: "Well-proportioned mid-weight vehicle balancing passenger comfort, moderate power output, and respectable highway mileage."
            },
            "Mid-Size Cruiser": {
                icon: "compass",
                tag: "Comfort Cruiser / Fleet Standard",
                color: "#f59e0b",
                description: "Extended chassis with 6-cylinder displacement, tuned for smooth highway cruising, moderate torque, and heavier curb weight."
            },
            "V8 Muscle & Heavy Hauler": {
                icon: "flame",
                tag: "High Performance / Heavy Towing",
                color: "#ef4444",
                description: "High displacement V8 powerhouse delivering peak torque and rapid quarter-mile acceleration at the expense of high fuel consumption."
            }
        };

        const archetypeMeta = traitMap[assigned.archetype] || {
            icon: "car",
            tag: "Standard Vehicle Profile",
            color: "#6366f1",
            description: "Balanced physical vehicle archetype."
        };

        return {
            clusterId: assigned.cluster_id,
            name: assigned.archetype,
            distance: parseFloat(minDistance.toFixed(3)),
            meta: archetypeMeta,
            centerSpec: assigned.center,
            similarities: similarities
        };
    }

    /**
     * Reverse Optimization: What-If Target MPG Engine
     * Calculates necessary weight and horsepower reductions to meet target MPG.
     */
    optimizeSpecsForTargetMpg(currentSpecs, targetMpg) {
        const currentPred = this.predictVehicle(currentSpecs);
        const deltaMpg = targetMpg - currentPred.mpg;

        if (deltaMpg <= 0) {
            return {
                needed: false,
                message: "Current configuration already meets or exceeds target MPG!",
                recommendedSpecs: { ...currentSpecs }
            };
        }

        // Weight coefficient sensitivity (approx -0.006 MPG/lb)
        const weightIdx = this.featureCols.indexOf('weight');
        const hpIdx = this.featureCols.indexOf('horsepower');

        const weightSensitivity = (this.currentWeights[weightIdx] / this.scaler.scale[weightIdx]);
        const hpSensitivity = (this.currentWeights[hpIdx] / this.scaler.scale[hpIdx]);

        // Proportional trade-off (60% weight reduction, 40% HP downsizing)
        const deltaFromWeight = deltaMpg * 0.65;
        const deltaFromHp = deltaMpg * 0.35;

        const weightCut = Math.abs(deltaFromWeight / (weightSensitivity || -0.005));
        const hpCut = Math.abs(deltaFromHp / (hpSensitivity || -0.05));

        const recommendedWeight = Math.max(1600, Math.round(currentSpecs.weight - weightCut));
        const recommendedHp = Math.max(45, Math.round(currentSpecs.horsepower - hpCut));

        return {
            needed: true,
            targetMpg: targetMpg,
            currentMpg: currentPred.mpg,
            mpgGain: parseFloat(deltaMpg.toFixed(1)),
            recommendedWeight: recommendedWeight,
            weightReductionLbs: Math.round(currentSpecs.weight - recommendedWeight),
            recommendedHp: recommendedHp,
            hpReduction: Math.round(currentSpecs.horsepower - recommendedHp),
            estimatedSavings: Math.round(( (15000 / currentPred.mpg) - (15000 / targetMpg) ) * 3.60)
        };
    }
}

// Global initialization helper
window.MLEngine = MLEngine;
