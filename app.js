/**
 * Automobile Efficiency Regressor & Vehicle Class Profiler
 * Main UI Controller & Reactive Event Bus
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Check Data Availability
    if (!window.AUTO_DATA) {
        console.error("Auto MPG Dataset failed to load.");
        return;
    }

    // 2. Initialize ML Core & Visualizer
    const engine = new MLEngine(window.AUTO_DATA);
    const visualizer = new MLVisualizer(engine);
    window.appEngine = engine;
    window.appVisualizer = visualizer;

    // 3. Application State
    const state = {
        activeModel: 'ridge',
        alpha: 1.0,
        l1Ratio: 0.5,
        specs: {
            cylinders: 4,
            displacement: 150,
            horsepower: 100,
            weight: 2800,
            acceleration: 15.5,
            model_year: 76,
            origin: 1
        },
        scatterX: 'weight',
        scatterY: 'mpg',
        dataset: {
            searchQuery: '',
            originFilter: 'all',
            archetypeFilter: 'all',
            sortColumn: 'mpg',
            sortAsc: false,
            currentPage: 1,
            pageSize: 10,
            filteredRecords: [...window.AUTO_DATA.records]
        }
    };

    // 4. Cache UI DOM Elements
    const elements = {
        // Sliders
        weightSlider: document.getElementById('weightSlider'),
        weightDisplay: document.getElementById('weightDisplay'),
        hpSlider: document.getElementById('hpSlider'),
        hpDisplay: document.getElementById('hpDisplay'),
        cylSlider: document.getElementById('cylSlider'),
        cylDisplay: document.getElementById('cylDisplay'),
        dispSlider: document.getElementById('dispSlider'),
        dispDisplay: document.getElementById('dispDisplay'),
        accelSlider: document.getElementById('accelSlider'),
        accelDisplay: document.getElementById('accelDisplay'),
        yearSlider: document.getElementById('yearSlider'),
        yearDisplay: document.getElementById('yearDisplay'),
        originSlider: document.getElementById('originSlider'),
        originDisplay: document.getElementById('originDisplay'),
        
        // Model & Hyperparameters
        modelButtons: document.querySelectorAll('.model-btn'),
        alphaSlider: document.getElementById('alphaSlider'),
        alphaValDisplay: document.getElementById('alphaValDisplay'),
        l1RatioGroup: document.getElementById('l1RatioGroup'),
        l1RatioSlider: document.getElementById('l1RatioSlider'),
        l1RatioDisplay: document.getElementById('l1RatioDisplay'),
        presetChips: document.querySelectorAll('.preset-chip[data-preset]'),

        // Telemetry & Gauge
        predMpgValue: document.getElementById('predMpgValue'),
        predCiValue: document.getElementById('predCiValue'),
        gradeBadge: document.getElementById('gradeBadge'),
        gaugeNeedleGroup: document.getElementById('gaugeNeedleGroup'),
        gaugeProgressArc: document.getElementById('gaugeProgressArc'),
        statAnnualCost: document.getElementById('statAnnualCost'),
        statAnnualSavings: document.getElementById('statAnnualSavings'),
        statCo2Tons: document.getElementById('statCo2Tons'),

        // Archetype Match Banner
        archetypeBanner: document.getElementById('archetypeBanner'),
        archetypeIconBox: document.getElementById('archetypeIconBox'),
        archetypeNameTag: document.getElementById('archetypeNameTag'),
        archetypeMatchScore: document.getElementById('archetypeMatchScore'),
        archetypeDesc: document.getElementById('archetypeDesc'),

        // Optimization
        targetMpgInput: document.getElementById('targetMpgInput'),
        calcOptimizeBtn: document.getElementById('calcOptimizeBtn'),
        targetOptimizeResult: document.getElementById('targetOptimizeResult'),

        // Scatter & Diagnostic Controls
        scatterXSelect: document.getElementById('scatterXSelect'),
        scatterYSelect: document.getElementById('scatterYSelect'),
        pathToggleRidge: document.getElementById('pathToggleRidge'),
        pathToggleLasso: document.getElementById('pathToggleLasso'),

        // Dataset Table
        carSearchInput: document.getElementById('carSearchInput'),
        originFilterSelect: document.getElementById('originFilterSelect'),
        archetypeFilterSelect: document.getElementById('archetypeFilterSelect'),
        recordCountDisplay: document.getElementById('recordCountDisplay'),
        vehiclesTableBody: document.getElementById('vehiclesTableBody'),
        currentPageDisplay: document.getElementById('currentPageDisplay'),
        totalPagesDisplay: document.getElementById('totalPagesDisplay'),
        prevPageBtn: document.getElementById('prevPageBtn'),
        nextPageBtn: document.getElementById('nextPageBtn'),
        tableHeaders: document.querySelectorAll('#vehiclesTable th[data-sort]'),

        // Modals
        deployGuideModal: document.getElementById('deployGuideModal'),
        openDeployGuideBtn: document.getElementById('openDeployGuideBtn'),
        closeDeployGuideBtn: document.getElementById('closeDeployGuideBtn'),
        closeGuideModalBtn2: document.getElementById('closeGuideModalBtn2')
    };

    // 5. Presets Mapping
    const presets = {
        corolla: { cylinders: 4, displacement: 97, horsepower: 75, weight: 2100, acceleration: 17.5, model_year: 78, origin: 3 },
        mustang: { cylinders: 8, displacement: 351, horsepower: 180, weight: 3800, acceleration: 10.5, model_year: 72, origin: 1 },
        sedan: { cylinders: 8, displacement: 307, horsepower: 130, weight: 3400, acceleration: 12.5, model_year: 70, origin: 1 },
        datsun: { cylinders: 6, displacement: 168, horsepower: 135, weight: 2600, acceleration: 13.0, model_year: 79, origin: 3 },
        vw: { cylinders: 4, displacement: 90, horsepower: 52, weight: 1950, acceleration: 21.0, model_year: 80, origin: 2 }
    };

    // 6. UI Sync and Reactive Pipeline
    function syncSlidersFromState() {
        elements.weightSlider.value = state.specs.weight;
        elements.weightDisplay.textContent = `${Number(state.specs.weight).toLocaleString()} lbs`;

        elements.hpSlider.value = state.specs.horsepower;
        elements.hpDisplay.textContent = `${state.specs.horsepower} HP`;

        elements.cylSlider.value = state.specs.cylinders;
        elements.cylDisplay.textContent = `${state.specs.cylinders} Cylinders`;

        elements.dispSlider.value = state.specs.displacement;
        elements.dispDisplay.textContent = `${state.specs.displacement} cu. in.`;

        elements.accelSlider.value = state.specs.acceleration;
        elements.accelDisplay.textContent = `${parseFloat(state.specs.acceleration).toFixed(1)} sec`;

        elements.yearSlider.value = state.specs.model_year;
        elements.yearDisplay.textContent = `19${state.specs.model_year}`;

        elements.originSlider.value = state.specs.origin;
        const originLabels = { 1: 'USA (1)', 2: 'Europe (2)', 3: 'Japan (3)' };
        elements.originDisplay.textContent = originLabels[state.specs.origin] || 'USA (1)';
    }

    function updateSimulation() {
        // Re-read slider values
        state.specs.weight = parseFloat(elements.weightSlider.value);
        state.specs.horsepower = parseFloat(elements.hpSlider.value);
        state.specs.cylinders = parseInt(elements.cylSlider.value);
        state.specs.displacement = parseFloat(elements.dispSlider.value);
        state.specs.acceleration = parseFloat(elements.accelSlider.value);
        state.specs.model_year = parseInt(elements.yearSlider.value);
        state.specs.origin = parseInt(elements.originSlider.value);

        syncSlidersFromState();

        // Perform ML Prediction
        const pred = engine.predictVehicle(state.specs);

        // Update Gauge Value
        elements.predMpgValue.textContent = pred.mpg.toFixed(1);
        elements.predCiValue.textContent = `95% CI: [${pred.ciLow} - ${pred.ciHigh}] MPG`;

        // Update Grade Badge
        elements.gradeBadge.textContent = pred.grade;
        elements.gradeBadge.style.color = pred.gradeColor;
        elements.gradeBadge.style.borderColor = pred.gradeColor;

        // Animate Speedometer Needle (-90deg at 5 MPG to +90deg at 50 MPG)
        const minMpg = 5;
        const maxMpg = 50;
        const clampedMpg = Math.max(minMpg, Math.min(maxMpg, pred.mpg));
        const percentage = (clampedMpg - minMpg) / (maxMpg - minMpg);
        const needleAngle = -90 + (percentage * 180);
        elements.gaugeNeedleGroup.style.transform = `rotate(${needleAngle}deg)`;

        // SVG Arc Dashoffset (total length ~ 283)
        const totalArcLength = 283;
        const offset = totalArcLength * (1 - percentage);
        elements.gaugeProgressArc.style.strokeDashoffset = offset;

        // Update Financial & Environmental Statistics
        elements.statAnnualCost.textContent = `$${pred.annualFuelCost.toLocaleString()}`;
        elements.statAnnualSavings.textContent = `${pred.annualSavingsVsAvg >= 0 ? '+' : '-'}$${Math.abs(pred.annualSavingsVsAvg).toLocaleString()}`;
        elements.statAnnualSavings.style.color = pred.annualSavingsVsAvg >= 0 ? 'var(--accent-emerald)' : 'var(--accent-crimson)';
        elements.statCo2Tons.innerHTML = `${pred.annualCo2Tons} <span style="font-size: 0.75rem;">Tons</span>`;

        // Update Archetype Match Banner
        const arch = pred.archetype;
        elements.archetypeNameTag.textContent = arch.name;
        elements.archetypeMatchScore.textContent = `${arch.similarities[0].similarity}%`;
        elements.archetypeDesc.textContent = arch.meta.description;
        elements.archetypeBanner.style.borderColor = arch.meta.color;

        const iconClassMap = {
            "leaf": "fa-leaf",
            "shield-check": "fa-shield-halved",
            "compass": "fa-compass",
            "flame": "fa-fire"
        };
        elements.archetypeIconBox.innerHTML = `<i class="fa-solid ${iconClassMap[arch.meta.icon] || 'fa-car'}"></i>`;
        elements.archetypeIconBox.style.color = arch.meta.color;
        elements.archetypeIconBox.style.backgroundColor = `${arch.meta.color}22`;

        // Update Visualizer Charts
        visualizer.renderFeatureImpact(pred.featureContributions);
        visualizer.renderArchetypeRadar(state.specs);
        visualizer.renderScatterPlot(state.scatterX, state.scatterY, state.specs);
    }

    // 7. Event Listeners: Sliders
    const allSliders = [
        elements.weightSlider, elements.hpSlider, elements.cylSlider,
        elements.dispSlider, elements.accelSlider, elements.yearSlider, elements.originSlider
    ];
    allSliders.forEach(slider => {
        slider.addEventListener('input', () => {
            updateSimulation();
        });
    });

    // 8. Event Listeners: Model Selection
    elements.modelButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.modelButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const model = btn.dataset.model;
            state.activeModel = model;

            // Show / hide L1 ratio slider for ElasticNet
            elements.l1RatioGroup.style.display = (model === 'elasticnet') ? 'block' : 'none';

            engine.trainActiveModel(state.activeModel, state.alpha, state.l1Ratio);
            visualizer.renderActualVsPredicted();
            updateSimulation();
        });
    });

    // 9. Event Listeners: Hyperparameter Sliders
    elements.alphaSlider.addEventListener('input', (e) => {
        state.alpha = parseFloat(e.target.value);
        elements.alphaValDisplay.textContent = state.alpha.toFixed(2);
        engine.trainActiveModel(state.activeModel, state.alpha, state.l1Ratio);
        visualizer.renderActualVsPredicted();
        updateSimulation();
    });

    elements.l1RatioSlider.addEventListener('input', (e) => {
        state.l1Ratio = parseFloat(e.target.value);
        elements.l1RatioDisplay.textContent = state.l1Ratio.toFixed(2);
        engine.trainActiveModel(state.activeModel, state.alpha, state.l1Ratio);
        visualizer.renderActualVsPredicted();
        updateSimulation();
    });

    // 10. Event Listeners: Preset Chips
    elements.presetChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const key = chip.dataset.preset;
            if (presets[key]) {
                state.specs = { ...presets[key] };
                syncSlidersFromState();
                updateSimulation();
            }
        });
    });

    // 11. Event Listeners: Reverse Optimization Advisor
    elements.calcOptimizeBtn.addEventListener('click', () => {
        const target = parseFloat(elements.targetMpgInput.value) || 35;
        const result = engine.optimizeSpecsForTargetMpg(state.specs, target);

        if (!result.needed) {
            elements.targetOptimizeResult.innerHTML = `
                <div style="color: var(--accent-emerald); font-weight: 600;">
                    <i class="fa-solid fa-circle-check"></i> ${result.message}
                </div>
            `;
        } else {
            elements.targetOptimizeResult.innerHTML = `
                To reach your target of <strong style="color: var(--primary-cyan);">${result.targetMpg} MPG</strong> (+$${result.mpgGain} MPG gain):
                <ul style="margin-top: 0.4rem; padding-left: 1.2rem; line-height: 1.6;">
                    <li>Reduce curb weight by <strong>${result.weightReductionLbs.toLocaleString()} lbs</strong> &rarr; Target: <strong>${result.recommendedWeight.toLocaleString()} lbs</strong></li>
                    <li>Downsize engine power by <strong>${result.hpReduction} HP</strong> &rarr; Target: <strong>${result.recommendedHp} HP</strong></li>
                    <li>Estimated Annual Fuel Savings: <strong style="color: var(--accent-emerald);">+$${result.estimatedSavings}/year</strong></li>
                </ul>
            `;
        }
    });

    // 12. Event Listeners: Scatter Axis Selectors
    elements.scatterXSelect.addEventListener('change', (e) => {
        state.scatterX = e.target.value;
        visualizer.renderScatterPlot(state.scatterX, state.scatterY, state.specs);
    });

    elements.scatterYSelect.addEventListener('change', (e) => {
        state.scatterY = e.target.value;
        visualizer.renderScatterPlot(state.scatterX, state.scatterY, state.specs);
    });

    // 13. Event Listeners: Regularization Path Toggle
    elements.pathToggleRidge.addEventListener('click', () => {
        elements.pathToggleRidge.classList.add('active');
        elements.pathToggleLasso.classList.remove('active');
        visualizer.renderRegularizationPath('ridge');
    });

    elements.pathToggleLasso.addEventListener('click', () => {
        elements.pathToggleLasso.classList.add('active');
        elements.pathToggleRidge.classList.remove('active');
        visualizer.renderRegularizationPath('lasso');
    });

    // 14. Dataset Table Logic & Filtering
    function filterAndRenderTable() {
        let list = [...window.AUTO_DATA.records];

        // Search query
        if (state.dataset.searchQuery.trim() !== '') {
            const q = state.dataset.searchQuery.toLowerCase();
            list = list.filter(r => r.name.toLowerCase().includes(q));
        }

        // Origin filter
        if (state.dataset.originFilter !== 'all') {
            const org = parseInt(state.dataset.originFilter);
            list = list.filter(r => r.origin === org);
        }

        // Archetype filter
        if (state.dataset.archetypeFilter !== 'all') {
            list = list.filter(r => r.archetype === state.dataset.archetypeFilter);
        }

        // Sorting
        const col = state.dataset.sortColumn;
        const asc = state.dataset.sortAsc;
        list.sort((a, b) => {
            let valA = a[col];
            let valB = b[col];
            if (typeof valA === 'string') {
                return asc ? valA.localeCompare(valB) : valB.localeCompare(valA);
            }
            return asc ? (valA - valB) : (valB - valA);
        });

        state.dataset.filteredRecords = list;
        state.dataset.currentPage = 1;
        renderTablePage();
    }

    function renderTablePage() {
        const list = state.dataset.filteredRecords;
        const total = list.length;
        const pageSize = state.dataset.pageSize;
        const totalPages = Math.ceil(total / pageSize) || 1;

        elements.recordCountDisplay.textContent = total;
        elements.totalPagesDisplay.textContent = totalPages;
        elements.currentPageDisplay.textContent = state.dataset.currentPage;

        elements.prevPageBtn.disabled = (state.dataset.currentPage <= 1);
        elements.nextPageBtn.disabled = (state.dataset.currentPage >= totalPages);

        const startIdx = (state.dataset.currentPage - 1) * pageSize;
        const pageItems = list.slice(startIdx, startIdx + pageSize);

        const originBadgeMap = {
            1: '<span style="color: #38bdf8;">USA</span>',
            2: '<span style="color: #f59e0b;">Europe</span>',
            3: '<span style="color: #10b981;">Japan</span>'
        };

        const archetypeClassMap = {
            "Eco Commuter": "badge-eco",
            "Balanced Daily Sedan": "badge-sedan",
            "Mid-Size Cruiser": "badge-cruiser",
            "V8 Muscle & Heavy Hauler": "badge-muscle"
        };

        elements.vehiclesTableBody.innerHTML = pageItems.map(item => `
            <tr>
                <td><strong>${item.name}</strong></td>
                <td><span style="font-family: var(--font-mono); font-weight: 700; color: #00f2fe;">${item.mpg.toFixed(1)}</span></td>
                <td>${item.cylinders}</td>
                <td>${item.horsepower}</td>
                <td>${item.weight.toLocaleString()}</td>
                <td>${item.acceleration.toFixed(1)}s</td>
                <td>'${item.model_year}</td>
                <td>${originBadgeMap[item.origin] || item.origin}</td>
                <td><span class="badge-tag ${archetypeClassMap[item.archetype] || 'badge-sedan'}">${item.archetype}</span></td>
                <td>
                    <button class="preset-chip load-row-btn" data-car='${JSON.stringify(item)}' style="padding: 0.2rem 0.5rem; font-size: 0.75rem;">
                        <i class="fa-solid fa-arrow-up-right-from-square"></i> Test
                    </button>
                </td>
            </tr>
        `).join('');

        // Attach load button handlers
        document.querySelectorAll('.load-row-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const carData = JSON.parse(btn.dataset.car);
                state.specs = {
                    cylinders: carData.cylinders,
                    displacement: carData.displacement,
                    horsepower: carData.horsepower,
                    weight: carData.weight,
                    acceleration: carData.acceleration,
                    model_year: carData.model_year,
                    origin: carData.origin
                };
                syncSlidersFromState();
                updateSimulation();
                document.getElementById('simulator').scrollIntoView({ behavior: 'smooth' });
            });
        });
    }

    elements.carSearchInput.addEventListener('input', (e) => {
        state.dataset.searchQuery = e.target.value;
        filterAndRenderTable();
    });

    elements.originFilterSelect.addEventListener('change', (e) => {
        state.dataset.originFilter = e.target.value;
        filterAndRenderTable();
    });

    elements.archetypeFilterSelect.addEventListener('change', (e) => {
        state.dataset.archetypeFilter = e.target.value;
        filterAndRenderTable();
    });

    elements.prevPageBtn.addEventListener('click', () => {
        if (state.dataset.currentPage > 1) {
            state.dataset.currentPage--;
            renderTablePage();
        }
    });

    elements.nextPageBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(state.dataset.filteredRecords.length / state.dataset.pageSize);
        if (state.dataset.currentPage < totalPages) {
            state.dataset.currentPage++;
            renderTablePage();
        }
    });

    elements.tableHeaders.forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (state.dataset.sortColumn === col) {
                state.dataset.sortAsc = !state.dataset.sortAsc;
            } else {
                state.dataset.sortColumn = col;
                state.dataset.sortAsc = true;
            }
            filterAndRenderTable();
        });
    });

    // 15. Deployment Modal Handlers
    elements.openDeployGuideBtn.addEventListener('click', () => {
        elements.deployGuideModal.classList.add('active');
    });

    elements.closeDeployGuideBtn.addEventListener('click', () => {
        elements.deployGuideModal.classList.remove('active');
    });

    elements.closeGuideModalBtn2.addEventListener('click', () => {
        elements.deployGuideModal.classList.remove('active');
    });

    elements.deployGuideModal.addEventListener('click', (e) => {
        if (e.target === elements.deployGuideModal) {
            elements.deployGuideModal.classList.remove('active');
        }
    });

    // 16. Initialize Application
    visualizer.initAllCharts();
    syncSlidersFromState();
    updateSimulation();
    filterAndRenderTable();
});
