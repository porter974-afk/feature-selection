/* FloodPulse AI – Interactive Browser Inference for Kaggle Flood Prediction */

const ALL_20_FEATURES = [
  // 氣象與水文 (Meteorology & Hydrology)
  { key: 'MonsoonIntensity', label: '季風降雨強度', cat: '氣象與水文', min: 0, max: 16, step: 1, base: 5 },
  { key: 'TopographyDrainage', label: '地形排水條件', cat: '氣象與水文', min: 0, max: 16, step: 1, base: 5 },
  { key: 'RiverManagement', label: '河川防洪治理', cat: '氣象與水文', min: 0, max: 16, step: 1, base: 5 },
  { key: 'Watersheds', label: '集水區保育治理', cat: '氣象與水文', min: 0, max: 16, step: 1, base: 5 },
  { key: 'ClimateChange', label: '氣候變遷極端化', cat: '氣象與水文', min: 0, max: 16, step: 1, base: 5 },

  // 水利工程與設施 (Infrastructure)
  { key: 'DamsQuality', label: '防洪大壩品質', cat: '水利工程與設施', min: 0, max: 16, step: 1, base: 5 },
  { key: 'Siltation', label: '河床淤積程度', cat: '水利工程與設施', min: 0, max: 16, step: 1, base: 5 },
  { key: 'DrainageSystems', label: '市區排水系統', cat: '水利工程與設施', min: 0, max: 16, step: 1, base: 5 },
  { key: 'DeterioratingInfrastructure', label: '設施老化損壞率', cat: '水利工程與設施', min: 0, max: 16, step: 1, base: 5 },

  // 環境土地與地質 (Environment & Geology)
  { key: 'Deforestation', label: '森林濫伐植被破壞', cat: '環境土地與地質', min: 0, max: 16, step: 1, base: 5 },
  { key: 'Urbanization', label: '都市化不透水率', cat: '環境土地與地質', min: 0, max: 16, step: 1, base: 5 },
  { key: 'WetlandLoss', label: '天然滯洪濕地流失', cat: '環境土地與地質', min: 0, max: 16, step: 1, base: 5 },
  { key: 'Landslides', label: '坡地崩塌土石流', cat: '環境土地與地質', min: 0, max: 16, step: 1, base: 5 },
  { key: 'CoastalVulnerability', label: '海岸風暴潮脆弱度', cat: '環境土地與地質', min: 0, max: 16, step: 1, base: 5 },
  { key: 'AgriculturalPractices', label: '農業水土保持不良', cat: '環境土地與地質', min: 0, max: 16, step: 1, base: 5 },

  // 人口與政策防救災 (Society & Planning)
  { key: 'PopulationScore', label: '淹水潛勢區人口密度', cat: '人口與政策防救災', min: 0, max: 16, step: 1, base: 5 },
  { key: 'Encroachments', label: '河道違規侵占違建', cat: '人口與政策防救災', min: 0, max: 16, step: 1, base: 5 },
  { key: 'IneffectiveDisasterPreparedness', label: '防災整備不足程度', cat: '人口與政策防救災', min: 0, max: 16, step: 1, base: 5 },
  { key: 'InadequatePlanning', label: '防洪規劃欠缺周全', cat: '人口與政策防救災', min: 0, max: 16, step: 1, base: 5 },
  { key: 'PoliticalFactors', label: '政策執行落實阻礙', cat: '人口與政策防救災', min: 0, max: 16, step: 1, base: 5 }
];

const BASE = Object.fromEntries(ALL_20_FEATURES.map(f => [f.key, f.base]));

const PRESETS = {
  highRisk: {
    MonsoonIntensity: 11, TopographyDrainage: 10, RiverManagement: 11, Deforestation: 11,
    Urbanization: 10, ClimateChange: 11, DamsQuality: 10, Siltation: 11,
    AgriculturalPractices: 10, Encroachments: 10, IneffectiveDisasterPreparedness: 11,
    DrainageSystems: 10, CoastalVulnerability: 11, Landslides: 11, Watersheds: 10,
    DeterioratingInfrastructure: 11, PopulationScore: 11, WetlandLoss: 10,
    InadequatePlanning: 11, PoliticalFactors: 10
  },
  monsoon: {
    MonsoonIntensity: 12, ClimateChange: 10, Landslides: 9, RiverManagement: 8,
    Siltation: 8, DamsQuality: 7, TopographyDrainage: 6, DeterioratingInfrastructure: 7,
    WetlandLoss: 6, DrainageSystems: 6
  },
  wellProtected: {
    MonsoonIntensity: 3, TopographyDrainage: 2, RiverManagement: 2, Deforestation: 2,
    Urbanization: 3, ClimateChange: 3, DamsQuality: 2, Siltation: 2,
    AgriculturalPractices: 2, Encroachments: 2, IneffectiveDisasterPreparedness: 2,
    DrainageSystems: 2, CoastalVulnerability: 2, Landslides: 2, Watersheds: 2,
    DeterioratingInfrastructure: 2, PopulationScore: 3, WetlandLoss: 2,
    InadequatePlanning: 2, PoliticalFactors: 2
  },
  median: { ...BASE }
};

const PHASES = [
  ['業務理解 (Business Understanding)',
   '全球極端氣候常態化，防洪治理與流域水利機關急需高準確、可歸因的洪災機率量化預測模型，以精準配置防災預算與工程防線。',
   ['預測目標：區域淹水或洪災爆發機率（FloodProbability：0.00 ~ 1.00）',
    '核心決策：辨識關鍵環境與基礎建設致災因子，避免單一盲目投資',
    '成功指標：測試集 R² ≥ 0.80、RMSE ≤ 0.03，並建立 95% 保形預測區間提供風險不確定度邊界']
  ],
  ['資料理解 (Data Understanding)',
   '採用 Kaggle Playground Series s4e5 官方真實資料集（train.csv），總計 1,117,957 筆樣本，包含完整 20 項環境、水利、地質與社經特徵。',
   ['嚴格驗證資料集具備 20 個原始輸入特徵（符合作業要求 >13 且 <40）',
    '徹底排除 id 編號與目標標籤 FloodProbability，絕不作為輸入特徵',
    '資料集完整無缺失值（0 Missing），目標變數呈接近常態分佈（中位數 0.505，範圍 0.285~0.725）']
  ],
  ['資料準備 (Data Preparation)',
   '以固定 random_state=42 將 1,117,957 筆資料劃分為 80% 訓練集（894,365 筆）與 20% 測試集（223,592 筆）。',
   ['防資料洩漏：所有統計檢驗與特徵選取均在訓練集內部獨立執行',
    '校準集切分：從訓練集再劃分 20% 校準子集（178,873 筆），專門用於計算 95% 保形預測區間分位數',
    '特徵型態檢驗：20 個特徵皆為數值型指標，無非數值污染']
  ],
  ['建模 (Modeling)',
   '使用 Scikit-Learn LinearRegression 建立完整多元線性迴歸模型，並搭配 RFE 淘汰挑選 10 項核心特徵進行比較。',
   ['全特徵 OLS：納入全部 20 項變數，捕捉全維度累積環境風險',
    'RFE 10 特徵：透過 Recursive Feature Elimination 迭代剔除權重最弱特徵，選出前 10 名核心變數',
    '保形預測校準：依非對稱殘差絕對值分位數，構建無分佈假設的 95% 預測區間']
  ],
  ['評估 (Evaluation)',
   '在 223,592 筆獨立測試集上評估模型性能，全維度 20 特徵大幅超越 RFE 10 特徵模型。',
   ['Full OLS 20 特徵：Test R² = 0.84488 | RMSE = 0.02008 | MAE = 0.01579',
    'RFE 10 特徵：Test R² = 0.37794 | RMSE = 0.04021 | MAE = 0.03247',
    '95% 預測區間實測覆蓋率：Full OLS 達到 94.89%（平均寬度 0.08012），RFE 10 達到 94.99%（平均寬度 0.15503）',
    '關鍵洞察：洪災風險是由 20 項因子共同線性疊加促成，削減特徵會直接導致預測能力腰斬']
  ],
  ['部署 (Deployment)',
   '沿用 GitHub Pages 靜態架構，將訓練完成之百萬資料模型參數、指標與 500 筆測試抽樣資料序列化為 JavaScript 供前端即時試算。',
   ['瀏覽器端零延遲即時推論，不需後端伺服器即可試算洪災機率',
    '提供 95% 預測區間上下限動態評估，直觀展現水利預警決策信心',
    '完整相容於專案中心（ml-projects.html），支援跨題材切換與深淺主題切換']
  ]
];

let architecture = 'full', impactChart, compareChart;

function estimate(v, type = architecture) {
  const isOptimal = type === 'optimal';
  const coef = isOptimal ? FLOOD_MODEL.rfeCoefficients : FLOOD_MODEL.fullCoefficients;
  let prediction = isOptimal ? FLOOD_MODEL.rfeIntercept : FLOOD_MODEL.fullIntercept;
  const parts = [['截距', prediction]];

  Object.entries(coef).forEach(([key, coefficient]) => {
    const val = Number(v[key] ?? BASE[key]);
    const amount = val * coefficient;
    prediction += amount;
    parts.push([key, amount]);
  });

  const clamped = Math.max(0.05, Math.min(0.95, prediction));
  const q95 = isOptimal ? FLOOD_MODEL.conformalPI.q95Rfe : FLOOD_MODEL.conformalPI.q95Full;
  const lower = Math.max(0, clamped - q95);
  const upper = Math.min(1, clamped + q95);

  return {
    prediction: clamped,
    lower,
    upper,
    q95,
    parts
  };
}

function getRiskCategory(prob) {
  if (prob >= 0.58) return { label: '極高洪災風險', cls: 'high' };
  if (prob >= 0.46) return { label: '中度警戒風險', cls: 'med' };
  return { label: '極低受災風險', cls: 'low' };
}

function values() {
  const v = { ...BASE };
  ALL_20_FEATURES.forEach(f => {
    const slider = document.getElementById(`${f.key}-slider`);
    if (slider) v[f.key] = Number(slider.value);
  });
  return v;
}

function renderControls() {
  const container = document.getElementById('controls');
  if (!container) return;

  const categories = ['氣象與水文', '水利工程與設施', '環境土地與地質', '人口與政策防救災'];
  let html = '';

  categories.forEach(cat => {
    html += `<div class="controls-category-title">${cat}</div>`;
    const catFeatures = ALL_20_FEATURES.filter(f => f.cat === cat);
    catFeatures.forEach(f => {
      const isRfe = FLOOD_MODEL.rfeFeatures.includes(f.key);
      html += `
        <div class="control-group">
          <div class="control-header">
            <label class="control-label" for="${f.key}-input">
              ${f.label} (${f.key})
              ${isRfe ? '<span class="model-badge" title="RFE 10 精選特徵">RFE 核心</span>' : ''}
              <span class="control-desc-sub">${FLOOD_MODEL.featureDescriptions[f.key] || ''}</span>
            </label>
            <input id="${f.key}-input" class="numeric-input" type="number" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.base}">
          </div>
          <input id="${f.key}-slider" class="styled-slider" type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.base}">
          <div class="slider-ticks">
            <span>低 (${f.min})</span>
            <span>中 (${(f.min + f.max)/2})</span>
            <span>極高 (${f.max})</span>
          </div>
        </div>
      `;
    });
  });

  container.innerHTML = html;

  ALL_20_FEATURES.forEach(f => {
    const slider = document.getElementById(`${f.key}-slider`);
    const input = document.getElementById(`${f.key}-input`);
    if (slider && input) {
      slider.addEventListener('input', () => { input.value = slider.value; update(); });
      input.addEventListener('input', () => { slider.value = input.value; update(); });
    }
  });
}

function update() {
  const v = values();
  const res = estimate(v);
  const full = estimate(v, 'full');
  const optimal = estimate(v, 'optimal');
  const risk = getRiskCategory(res.prediction);

  const probPercent = (res.prediction * 100).toFixed(2) + '%';
  const probValEl = document.getElementById('predicted-prob-val');
  if (probValEl) probValEl.innerHTML = `${probPercent} <span class="risk-pill ${risk.cls}">${risk.label}</span>`;

  const typeLabel = document.getElementById('model-type-label');
  if (typeLabel) typeLabel.textContent = architecture === 'full' ? 'Full OLS (20 特徵)' : 'RFE Optimal (10 特徵)';

  const countLabel = document.getElementById('model-features-count');
  if (countLabel) countLabel.textContent = architecture === 'full' ? '20 項原始特徵' : '10 項 RFE 特徵';

  const ciVal = document.getElementById('predicted-ci-val');
  if (ciVal) {
    const lo = (res.lower * 100).toFixed(1) + '%';
    const up = (res.upper * 100).toFixed(1) + '%';
    const half = (res.q95 * 100).toFixed(2) + '%';
    ciVal.textContent = `[${lo} ~ ${up}] (±${half})`;
  }

  // Top risk contributors
  const namedParts = res.parts.filter(([name]) => name !== '截距')
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  const top6 = namedParts.slice(0, 6);
  const attrContainer = document.getElementById('attr-items-container');
  if (attrContainer) {
    attrContainer.innerHTML = top6.map(([name, amount]) => {
      const desc = FLOOD_MODEL.featureDescriptions[name] || name;
      const pct = (amount * 100).toFixed(2);
      return `
        <div class="attr-item">
          <span class="attr-name">
            ${desc}
            <small>特徵值: ${v[name]} × 權重: ${architecture === 'full' ? FLOOD_MODEL.fullCoefficients[name]?.toFixed(4) : FLOOD_MODEL.rfeCoefficients[name]?.toFixed(4)}</small>
          </span>
          <b class="attr-contrib ${amount >= 0 ? 'positive' : 'negative'}">
            +${pct}%
          </b>
        </div>
      `;
    }).join('');
  }

  // Update charts
  if (impactChart) {
    impactChart.data.labels = top6.map(x => x[0]);
    impactChart.data.datasets[0].data = top6.map(x => +(x[1] * 100).toFixed(3));
    impactChart.update('none');
  }

  if (compareChart) {
    compareChart.data.datasets[0].data = [
      +(full.prediction * 100).toFixed(2),
      +(optimal.prediction * 100).toFixed(2)
    ];
    compareChart.update('none');
  }
}

function initCharts() {
  const color = document.body.dataset.theme === 'light' ? '#475569' : '#94a3b8';

  const impactCtx = document.getElementById('featureImpactChart');
  if (impactCtx) {
    impactChart = new Chart(impactCtx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: ['#06b6d4', '#38bdf8', '#818cf8', '#6366f1', '#a855f7', '#ec4899'],
          borderRadius: 5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: c => `邊際洪災風險貢獻: +${c.raw}%` }
          }
        },
        scales: {
          x: { ticks: { color, font: { size: 9 } }, grid: { display: false } },
          y: { ticks: { color, callback: x => x + '%' }, grid: { color: 'rgba(148,163,184,.12)' } }
        }
      }
    });
  }

  const compareCtx = document.getElementById('modelCompareChart');
  if (compareCtx) {
    compareChart = new Chart(compareCtx, {
      type: 'bar',
      data: {
        labels: ['Full OLS (20 特徵)', 'RFE Optimal (10 特徵)'],
        datasets: [{
          data: [],
          backgroundColor: ['#06b6d4', '#10b981'],
          borderRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: c => `預測洪災機率: ${c.raw}%` }
          }
        },
        scales: {
          x: { ticks: { color }, grid: { display: false } },
          y: { min: 0, max: 100, ticks: { color, callback: x => x + '%' }, grid: { color: 'rgba(148,163,184,.12)' } }
        }
      }
    });
  }
}

function renderPhases(active = 0) {
  const pillsEl = document.getElementById('crisp-pills');
  const cardEl = document.getElementById('crisp-card');
  if (!pillsEl || !cardEl) return;

  pillsEl.innerHTML = PHASES.map(([title], i) => `
    <button class="crisp-pill ${i === active ? 'active' : ''}" data-phase="${i}">
      <span class="pill-num">${i + 1}</span>
      <span class="pill-text">${title.split(' ')[0]}</span>
    </button>
  `).join('');

  const [title, desc, points] = PHASES[active];
  cardEl.innerHTML = `
    <span class="crisp-badge-phase">Phase ${active + 1} · CRISP-DM 流程</span>
    <h3 class="crisp-detail-title">${title}</h3>
    <div class="crisp-detail-grid">
      <div class="crisp-text">
        <p>${desc}</p>
        <ul class="crisp-keypoints">${points.map(x => `<li>${x}</li>`).join('')}</ul>
      </div>
      <pre class="crisp-code-preview">${active === 3 ? "Scikit-Learn RFE 特徵淘汰：\n20 特徵選取 10 核心特徵\nFull Test R²: 0.8449 | RMSE: 0.0201\nRFE Test R²:  0.3779 | RMSE: 0.0402\n95% PI 覆蓋率: 94.89% (±0.0401)" : "Kaggle Playground Series s4e5\n1,117,957 筆海量樣本 · 20 原始特徵\n目標：FloodProbability (0.285~0.725)"}</pre>
    </div>
  `;

  document.querySelectorAll('.crisp-pill').forEach(btn => {
    btn.onclick = () => renderPhases(Number(btn.dataset.phase));
  });
}

function renderTable() {
  const term = document.getElementById('dataset-search')?.value.trim().toLowerCase() || '';
  const filter = document.getElementById('risk-filter')?.value || 'all';
  const tableBody = document.getElementById('table-body');
  const countEl = document.getElementById('table-record-count');

  if (!tableBody || typeof FLOOD_DATA === 'undefined') return;

  const rows = FLOOD_DATA.filter(row => {
    const p = row.FloodProbability;
    const matchFilter = (filter === 'all') ||
      (filter === 'low' && p < 0.46) ||
      (filter === 'mid' && p >= 0.46 && p <= 0.58) ||
      (filter === 'high' && p > 0.58);
    const matchTerm = !term || Object.values(row).join(' ').toLowerCase().includes(term);
    return matchFilter && matchTerm;
  });

  tableBody.innerHTML = rows.map(row => {
    const idx = FLOOD_DATA.indexOf(row);
    const predFull = estimate(row, 'full').prediction;
    const predPercent = (predFull * 100).toFixed(1) + '%';
    const actPercent = (row.FloodProbability * 100).toFixed(1) + '%';
    const err = row.FloodProbability - predFull;
    const errPercent = (err >= 0 ? '+' : '') + (err * 100).toFixed(2) + '%';

    return `
      <tr data-index="${idx}">
        <td>#${idx + 1}</td>
        <td>${row.MonsoonIntensity}</td>
        <td>${row.RiverManagement}</td>
        <td>${row.Deforestation}</td>
        <td>${row.Urbanization}</td>
        <td>${row.DamsQuality}</td>
        <td>${row.DrainageSystems}</td>
        <td>${row.Landslides}</td>
        <td class="font-mono font-bold">${actPercent}</td>
        <td class="font-mono text-cyan">${predPercent}</td>
        <td class="font-mono ${err >= 0 ? 'text-rose' : 'text-emerald'}">${errPercent}</td>
        <td><button class="btn-table-action">帶入</button></td>
      </tr>
    `;
  }).join('');

  if (countEl) countEl.textContent = `顯示 ${rows.length} / ${FLOOD_DATA.length} 筆測試抽樣紀錄 (總樣本 1,117,957 筆)`;

  document.querySelectorAll('#table-body tr').forEach(tr => {
    tr.onclick = () => loadRow(FLOOD_DATA[tr.dataset.index]);
  });
}

function loadRow(row) {
  ALL_20_FEATURES.forEach(f => {
    const slider = document.getElementById(`${f.key}-slider`);
    const input = document.getElementById(`${f.key}-input`);
    if (slider && input && row[f.key] !== undefined) {
      slider.value = row[f.key];
      input.value = row[f.key];
    }
  });
  update();
  document.getElementById('simulator-section')?.scrollIntoView({ behavior: 'smooth' });
}

document.addEventListener('DOMContentLoaded', () => {
  renderControls();
  initCharts();
  renderPhases();
  renderTable();
  update();

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.onclick = () => {
      const p = PRESETS[btn.dataset.preset];
      if (p) {
        loadRow({ ...BASE, ...p });
        document.querySelectorAll('.preset-btn').forEach(x => x.classList.toggle('active', x === btn));
      }
    };
  });

  document.querySelectorAll('.model-arch-btn').forEach(btn => {
    btn.onclick = () => {
      architecture = btn.dataset.arch;
      document.querySelectorAll('.model-arch-btn').forEach(x => x.classList.toggle('active', x === btn));
      update();
    };
  });

  document.getElementById('btn-reset')?.addEventListener('click', () => {
    loadRow(BASE);
    document.querySelectorAll('.preset-btn').forEach(x => x.classList.remove('active'));
  });

  document.getElementById('btn-randomize')?.addEventListener('click', () => {
    if (typeof FLOOD_DATA !== 'undefined' && FLOOD_DATA.length) {
      const randRow = FLOOD_DATA[Math.floor(Math.random() * FLOOD_DATA.length)];
      loadRow(randRow);
    }
  });

  document.getElementById('dataset-search')?.addEventListener('input', renderTable);
  document.getElementById('risk-filter')?.addEventListener('change', renderTable);

  const themeBtn = document.getElementById('theme-toggle-btn');
  if (themeBtn) {
    themeBtn.onclick = () => {
      document.body.dataset.theme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
      themeBtn.textContent = document.body.dataset.theme === 'dark' ? '☼' : '☾';
    };
  }
});
