// HERE AND NOW AI - ML Playground logic
const regCtx = document.getElementById('regChart').getContext('2d');
const classCtx = document.getElementById('classChart').getContext('2d');

let regPoints = [];
let classPoints = [];
let regChart, classChart;

let currentTheta = [0, 0]; // For Gradient Descent [bias, weight]

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab + '-tab').classList.add('active');
    });
});

// --- Regression Logic ---

function generateRegData() {
    regPoints = [];
    const trueBias = 4;
    const trueWeight = 3;
    const modelType = document.getElementById('model-type').value;
    for (let i = 0; i < 50; i++) {
        const x = Math.random() * 5;
        let y = trueBias + (trueWeight * x);
        if (modelType === 'polynomial') {
            y += 0.5 * Math.pow(x, 2); // Add curve
        }
        const noise = (Math.random() - 0.5) * 3;
        regPoints.push({ x: x, y: y + noise });
    }
    currentTheta = [0, 0]; // Reset GD
}

function calculateMetrics(preds) {
    let mae = 0, mse = 0;
    let ssRes = 0, ssTot = 0;
    let sumY = regPoints.reduce((acc, p) => acc + p.y, 0);
    let meanY = sumY / regPoints.length;

    regPoints.forEach((p, i) => {
        const err = Math.abs(p.y - preds[i]);
        mae += err;
        mse += Math.pow(err, 2);
        ssRes += Math.pow(p.y - preds[i], 2);
        ssTot += Math.pow(p.y - meanY, 2);
    });

    const n = regPoints.length;
    mae /= n;
    mse /= n;
    const r2 = 1 - (ssRes / ssTot);

    document.getElementById('mae-val').innerText = mae.toFixed(2);
    document.getElementById('mse-val').innerText = mse.toFixed(2);
    document.getElementById('rmse-val').innerText = Math.sqrt(mse).toFixed(2);
    document.getElementById('r2-val').innerText = r2.toFixed(2);
}

function updateRegChart() {
    const modelType = document.getElementById('model-type').value;
    const solveMethod = document.getElementById('solve-method').value;
    
    let linePoints = [];
    let preds = [];

    if (modelType === 'linear' && solveMethod === 'normal') {
        const n = regPoints.length;
        let sX = 0, sY = 0, sXY = 0, sX2 = 0;
        regPoints.forEach(p => {
            sX += p.x; sY += p.y; sXY += p.x * p.y; sX2 += p.x * p.x;
        });
        const w = (n * sXY - sX * sY) / (n * sX2 - sX * sX);
        const b = (sY - w * sX) / n;
        
        for (let x = 0; x <= 5; x += 0.1) linePoints.push({ x: x, y: b + w * x });
        regPoints.forEach(p => preds.push(b + w * p.x));
        
        document.getElementById('coef-interpretation').innerText = 
            `Money = ${b.toFixed(2)} (Starting Cash) + ${w.toFixed(2)} × Lemons. Every extra lemon adds $${w.toFixed(2)}!`;
    } else if (modelType === 'polynomial') {
        const deg = parseInt(document.getElementById('poly-degree').value);
        // Simple parabolic fit simulation for playground
        let a = 4, b = 1, c = 1.2; 
        for (let x = 0; x <= 5; x += 0.1) linePoints.push({ x: x, y: a + b*x + c*Math.pow(x, 2) });
        regPoints.forEach(p => preds.push(a + b*p.x + c*Math.pow(p.x, 2)));
        document.getElementById('coef-interpretation').innerText = 
            "The robot is using 📐 Squared Lemons (Lemons²) to bend the line and fit the curve!";
    }

    regChart.data.datasets[1].data = linePoints;
    regChart.data.datasets[1].label = modelType === 'linear' ? 'Linear (The Ruler)' : 'Polynomial (The Pipe Cleaner)';
    regChart.update();
    calculateMetrics(preds);
}

function generateClassData() {
    classPoints = [];
    for (let i = 0; i < 40; i++) {
        const size = Math.random() * 10;
        const prob = 1 / (1 + Math.exp(-(size - 5)));
        const isOrange = Math.random() < prob ? 1 : 0;
        classPoints.push({ x: size, y: isOrange });
    }
}

function updateClassChart() {
    const threshold = parseFloat(document.getElementById('threshold').value);
    document.getElementById('threshold-val').innerText = threshold;

    let tp=0, fp=0, fn=0, tn=0;
    classPoints.forEach(p => {
        const prob = 1 / (1 + Math.exp(-(p.x - 5))); 
        const pred = prob >= threshold ? 1 : 0;
        if (pred === 1 && p.y === 1) tp++;
        else if (pred === 1 && p.y === 0) fp++;
        else if (pred === 0 && p.y === 1) fn++;
        else tn++;
    });

    document.getElementById('tp-val').innerText = tp;
    document.getElementById('fp-val').innerText = fp;
    document.getElementById('fn-val').innerText = fn;
    document.getElementById('tn-val').innerText = tn;

    const acc = (tp + tn) / classPoints.length;
    const prec = tp / (tp + fp) || 0;
    const rec = tp / (tp + fn) || 0;
    const f1 = 2 * (prec * rec) / (prec + rec) || 0;

    document.getElementById('acc-val').innerText = (acc * 100).toFixed(1) + '%';
    document.getElementById('prec-val').innerText = (prec * 100).toFixed(1) + '%';
    document.getElementById('rec-val').innerText = (rec * 100).toFixed(1) + '%';
    document.getElementById('f1-val').innerText = f1.toFixed(2);

    classChart.data.datasets[0].data = classPoints.map(p => ({ x: p.x, y: p.y }));
    classChart.update();
}

function init() {
    generateRegData();
    generateClassData();

    regChart = new Chart(regCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Real Data',
                data: regPoints,
                backgroundColor: '#004040'
            }, {
                label: 'Fit',
                data: [],
                type: 'line',
                borderColor: '#FFDF00',
                borderWidth: 4,
                fill: false,
                pointRadius: 0
            }]
        },
        options: { scales: { x: { min: 0, max: 5 }, y: { min: 0, max: 30 } } }
    });

    classChart = new Chart(classCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Fruits',
                data: [],
                backgroundColor: (context) => {
                    const index = context.dataIndex;
                    if (index < 0) return '#000';
                    return classPoints[index]?.y === 1 ? '#ffa500' : '#ffff00';
                },
                pointRadius: 8
            }]
        },
        options: {
            scales: { 
                y: { ticks: { callback: v => v === 1 ? 'Orange' : 'Lemon' }, min: -0.2, max: 1.2 },
                x: { title: { display: true, text: 'Size (cm)' } }
            }
        }
    });

    updateRegChart();
    updateClassChart();
}

document.getElementById('model-type').addEventListener('change', (e) => {
    document.getElementById('poly-degree-container').style.display = e.target.value === 'polynomial' ? 'block' : 'none';
    generateRegData();
    updateRegChart();
});

document.getElementById('threshold').addEventListener('input', updateClassChart);
document.getElementById('reset-btn').addEventListener('click', () => { generateRegData(); updateRegChart(); });
document.getElementById('solve-btn').addEventListener('click', updateRegChart);

init();
