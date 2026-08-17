/**
 * Automobile Efficiency Regressor & Vehicle Class Profiler
 * Interactive Charting & Data Visualization Module (Chart.js Integration)
 */

class MLVisualizer {
    constructor(engine) {
        this.engine = engine;
        this.charts = {};
        this.colorPalette = {
            eco: 'rgba(16, 185, 129, 0.85)',
            sedan: 'rgba(6, 182, 212, 0.85)',
            cruiser: 'rgba(245, 158, 11, 0.85)',
            muscle: 'rgba(239, 68, 68, 0.85)',
            custom: '#00f2fe',
            grid: 'rgba(255, 255, 255, 0.08)',
            text: '#94a3b8',
            accent: '#38bdf8'
        };
    }

    /**
     * Initialize all interactive charts on the page
     */
    initAllCharts() {
        this.renderArchetypeRadar();
        this.renderScatterPlot('weight', 'mpg');
        this.renderRegularizationPath('ridge');
        this.renderFeatureImpact([]);
        this.renderActualVsPredicted();
    }

    /**
     * 1. Radar Profile Chart (Vehicle Physical Archetypes)
     */
    renderArchetypeRadar(customSpecs = null) {
        const ctx = document.getElementById('archetypeRadarChart');
        if (!ctx) return;

        const labels = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year'];
        const archetypes = this.engine.centroids;

        // Normalize centroid physical stats to [0-100] scale for visual radar balance
        const maxVals = [8, 455, 230, 5140, 24.8, 82];
        const minVals = [3, 68, 46, 1613, 8.0, 70];

        const normalize = (spec) => {
            return [
                ((spec.cylinders - minVals[0]) / (maxVals[0] - minVals[0])) * 100,
                ((spec.displacement - minVals[1]) / (maxVals[1] - minVals[1])) * 100,
                ((spec.horsepower - minVals[2]) / (maxVals[2] - minVals[2])) * 100,
                ((spec.weight - minVals[3]) / (maxVals[3] - minVals[3])) * 100,
                ((spec.acceleration - minVals[4]) / (maxVals[4] - minVals[4])) * 100,
                ((spec.model_year - minVals[5]) / (maxVals[5] - minVals[5])) * 100
            ];
        };

        const colors = [
            { border: '#10b981', bg: 'rgba(16, 185, 129, 0.15)' },
            { border: '#06b6d4', bg: 'rgba(6, 182, 212, 0.15)' },
            { border: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)' },
            { border: '#ef4444', bg: 'rgba(239, 68, 68, 0.15)' }
        ];

        const datasets = archetypes.map((arch, i) => ({
            label: arch.archetype,
            data: normalize(arch.center),
            borderColor: colors[i % colors.length].border,
            backgroundColor: colors[i % colors.length].bg,
            borderWidth: 2,
            pointBackgroundColor: colors[i % colors.length].border,
            pointRadius: 3
        }));

        if (customSpecs) {
            datasets.push({
                label: 'Your Vehicle Config',
                data: normalize(customSpecs),
                borderColor: '#00f2fe',
                backgroundColor: 'rgba(0, 242, 254, 0.35)',
                borderWidth: 3,
                pointBackgroundColor: '#00f2fe',
                pointHoverRadius: 7,
                pointRadius: 5
            });
        }

        if (this.charts.radar) {
            this.charts.radar.data.datasets = datasets;
            this.charts.radar.update('none');
            return;
        }

        this.charts.radar = new Chart(ctx, {
            type: 'radar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#cbd5e1', font: { family: 'Outfit, sans-serif', size: 12 }, padding: 15 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${Math.round(ctx.raw)}% profile intensity`
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        pointLabels: { color: '#cbd5e1', font: { family: 'Outfit, sans-serif', size: 12, weight: '500' } },
                        ticks: { display: false, min: 0, max: 100 }
                    }
                }
            }
        });
    }

    /**
     * 2. Scatter Plot: Dataset Distribution & Clusters
     */
    renderScatterPlot(xAxis = 'weight', yAxis = 'mpg', customPoint = null) {
        const ctx = document.getElementById('clusterScatterChart');
        if (!ctx) return;

        const clusterGroups = {
            0: { label: 'Eco Commuter', color: '#10b981', data: [] },
            1: { label: 'Balanced Daily Sedan', color: '#06b6d4', data: [] },
            2: { label: 'Mid-Size Cruiser', color: '#f59e0b', data: [] },
            3: { label: 'V8 Muscle & Heavy Hauler', color: '#ef4444', data: [] }
        };

        this.engine.records.forEach(r => {
            const cId = r.cluster;
            if (clusterGroups[cId]) {
                clusterGroups[cId].data.push({
                    x: r[xAxis],
                    y: r[yAxis],
                    name: r.name,
                    mpg: r.mpg,
                    hp: r.horsepower,
                    weight: r.weight,
                    cyl: r.cylinders
                });
            }
        });

        const datasets = Object.values(clusterGroups).map(g => ({
            label: g.label,
            data: g.data,
            backgroundColor: g.color,
            borderColor: 'rgba(0,0,0,0.3)',
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 7
        }));

        if (customPoint) {
            datasets.push({
                label: '⚡ Current Custom Vehicle',
                data: [{
                    x: customPoint[xAxis],
                    y: customPoint[yAxis],
                    name: 'Custom Configuration'
                }],
                backgroundColor: '#ffffff',
                borderColor: '#00f2fe',
                borderWidth: 3,
                pointRadius: 9,
                pointHoverRadius: 11
            });
        }

        const xTitles = {
            weight: 'Vehicle Weight (lbs)',
            horsepower: 'Engine Horsepower (HP)',
            displacement: 'Engine Displacement (cu. in.)',
            acceleration: '0-60 MPH Acceleration Time (seconds)'
        };

        const yTitles = {
            mpg: 'Fuel Efficiency (Miles Per Gallon)',
            horsepower: 'Horsepower (HP)'
        };

        if (this.charts.scatter) {
            this.charts.scatter.data.datasets = datasets;
            this.charts.scatter.options.scales.x.title.text = xTitles[xAxis] || xAxis;
            this.charts.scatter.options.scales.y.title.text = yTitles[yAxis] || yAxis;
            this.charts.scatter.update('none');
            return;
        }

        this.charts.scatter = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 }, boxWidth: 12 }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        borderColor: '#38bdf8',
                        borderWidth: 1,
                        titleFont: { family: 'Outfit', weight: 'bold' },
                        callbacks: {
                            label: (ctx) => {
                                const pt = ctx.raw;
                                return [
                                    `${pt.name}`,
                                    `MPG: ${pt.y || pt.mpg} | HP: ${pt.hp || '-'} | Wt: ${pt.weight || '-'} lbs`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: xTitles[xAxis] || xAxis, color: '#38bdf8', font: { size: 12, weight: '600' } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: yTitles[yAxis] || yAxis, color: '#38bdf8', font: { size: 12, weight: '600' } }
                    }
                }
            }
        });
    }

    /**
     * 3. Regularization Shrinkage Path (Ridge vs Lasso)
     */
    renderRegularizationPath(modelType = 'ridge') {
        const ctx = document.getElementById('regularizationPathChart');
        if (!ctx) return;

        const pathData = this.engine.paths;
        if (!pathData) return;

        const alphas = pathData.alphas;
        const coefPaths = modelType === 'lasso' ? pathData.lasso : pathData.ridge;

        const featureColors = {
            cylinders: '#ef4444',
            displacement: '#f97316',
            horsepower: '#f59e0b',
            weight: '#10b981',
            acceleration: '#06b6d4',
            model_year: '#8b5cf6',
            origin: '#ec4899'
        };

        const datasets = Object.keys(coefPaths).map(feature => ({
            label: feature.replace('_', ' ').toUpperCase(),
            data: coefPaths[feature].map((val, idx) => ({ x: alphas[idx], y: val })),
            borderColor: featureColors[feature] || '#cbd5e1',
            borderWidth: 2.2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.3
        }));

        if (this.charts.path) {
            this.charts.path.data.datasets = datasets;
            this.charts.path.options.plugins.title.text = `${modelType.toUpperCase()} Regularization Path: Feature Shrinkage vs Penalty \u03B1`;
            this.charts.path.update();
            return;
        }

        this.charts.path = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `${modelType.toUpperCase()} Regularization Path: Feature Shrinkage vs Penalty \u03B1`,
                        color: '#f1f5f9',
                        font: { family: 'Outfit', size: 14, weight: '600' }
                    },
                    legend: {
                        position: 'bottom',
                        labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 }, boxWidth: 10, padding: 12 }
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => `Penalty \u03B1 = ${items[0].parsed.x.toFixed(3)}`,
                            label: (item) => `${item.dataset.label}: \u03B2 = ${item.parsed.y.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'logarithmic',
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Regularization Parameter \u03B1 (Log Scale)', color: '#38bdf8', font: { size: 12 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Standardized Coefficient Weight (\u03B2)', color: '#38bdf8', font: { size: 12 } }
                    }
                }
            }
        });
    }

    /**
     * 4. Feature Impact Breakdown (Waterfall / Bar)
     */
    renderFeatureImpact(contributions = []) {
        const ctx = document.getElementById('featureImpactChart');
        if (!ctx) return;

        const sorted = [...contributions].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
        const labels = sorted.map(c => c.feature.replace('_', ' ').toUpperCase());
        const data = sorted.map(c => c.impact);
        const bgColors = data.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)');
        const borderColors = data.map(v => v >= 0 ? '#10b981' : '#ef4444');

        if (this.charts.impact) {
            this.charts.impact.data.labels = labels;
            this.charts.impact.data.datasets[0].data = data;
            this.charts.impact.data.datasets[0].backgroundColor = bgColors;
            this.charts.impact.data.datasets[0].borderColor = borderColors;
            this.charts.impact.update('none');
            return;
        }

        this.charts.impact = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'MPG Impact (\u0394 Miles/Gal)',
                    data: data,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.raw >= 0 ? '+' : ''}${ctx.raw} MPG impact on efficiency`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Net MPG Impact (\u0394)', color: '#38bdf8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#cbd5e1', font: { family: 'Outfit', weight: '500' } }
                    }
                }
            }
        });
    }

    /**
     * 5. Actual vs Predicted Scatter Plot
     */
    renderActualVsPredicted() {
        const ctx = document.getElementById('actualVsPredChart');
        if (!ctx) return;

        const evalResult = this.engine.evaluateModel();
        const predictions = evalResult.predictions;
        const actuals = this.engine.y;

        const points = [];
        for (let i = 0; i < actuals.length; i++) {
            points.push({ x: actuals[i], y: predictions[i] });
        }

        // 45-degree diagonal line points
        const minMpg = 8;
        const maxMpg = 48;
        const line45 = [
            { x: minMpg, y: minMpg },
            { x: maxMpg, y: maxMpg }
        ];

        const datasets = [
            {
                type: 'scatter',
                label: 'Vehicles (Actual vs Predicted)',
                data: points,
                backgroundColor: 'rgba(56, 189, 248, 0.65)',
                borderColor: '#0284c7',
                borderWidth: 1,
                pointRadius: 3.5,
                pointHoverRadius: 6
            },
            {
                type: 'line',
                label: 'Perfect Fit (y = x)',
                data: line45,
                borderColor: '#10b981',
                borderWidth: 2,
                borderDash: [6, 4],
                pointRadius: 0,
                fill: false
            }
        ];

        if (this.charts.actualVsPred) {
            this.charts.actualVsPred.data.datasets = datasets;
            this.charts.actualVsPred.update();
            return;
        }

        this.charts.actualVsPred = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Actual: ${ctx.parsed.x} MPG | Predicted: ${ctx.parsed.y.toFixed(1)} MPG (Err: ${(ctx.parsed.y - ctx.parsed.x).toFixed(1)})`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Actual UCI Ground Truth MPG', color: '#38bdf8' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.07)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Model Predicted MPG', color: '#38bdf8' }
                    }
                }
            }
        });
    }
}

window.MLVisualizer = MLVisualizer;
