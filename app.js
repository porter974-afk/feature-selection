/**
 * 50 Startups ML Platform - Client Application
 * Powered by Scikit-Learn Regression Coefficients & CRISP-DM Standard
 */

// ============================================================================
// 1. Model Parameters (Extracted directly from Scikit-Learn Pipeline)
// ============================================================================
const MODEL_CONFIG = {
  intercept: 54028.03959405866,
  coef: {
    rd: 0.8056300637169751,
    admin: -0.06878788228041122,
    marketing: 0.029855442922038122,
    florida: 938.7930061484041,
    new_york: 6.987759966912739,
    california: 0.0 // Baseline
  }
};

// ============================================================================
// 2. CRISP-DM 6 Phases Content Data
// ============================================================================
const CRISP_PHASES = {
  1: {
    badge: "Phase 1 • 業務目標與效益定錨",
    title: "商業理解 (Business Understanding)",
    desc: "風險投資（VC）機構與企業決策層的核心問題：資金究竟投在哪個部門效益最高？地理營運州別是否提供結構性利潤優勢？",
    points: [
      "核心商業命題：量化研發、行政與行銷三項費用的投資報酬率（ROI）。",
      "成功準則：判定係數 R² ≥ 0.85，殘差滿足古典常態性與無自相關。",
      "決策支援應用：建置新創投資標的自動化審查與利潤估算管線。"
    ],
    code: `# 商業目標定義
# 目標變數：Profit（新創公司年淨利潤）
# 自變數：R&D Spend, Administration, Marketing Spend, State
# 成功檢驗指標：R2 >= 0.85, 邊際 ROI 統計顯著性 (p < 0.05)`
  },
  2: {
    badge: "Phase 2 • 資料型態、分佈與多重共線性檢定",
    title: "資料理解 (Data Understanding)",
    desc: "對 50 家美國新創公司歷史資料進行嚴謹的探索性資料分析（EDA）與變異數膨脹因子（VIF）多重共線性檢定。",
    points: [
      "資料品質檢視：50 筆紀錄，0 筆缺失值（Missing Values = 0）。",
      "極值特例發現：2 家新創研發投入為 $0，3 家行銷推廣為 $0。",
      "皮爾森相關分析：研發支出與利潤高度正相關 (r = 0.973)；行銷與研發有輕微共線性 (r = 0.724)。",
      "VIF 檢定確認：研發 VIF=2.47，行銷 VIF=2.33，均小於 5.0 門檻，可安全建立模型。"
    ],
    code: `import pandas as pd
df = pd.read_csv('50_Startups.csv')

# 核心相關矩陣
corr = df[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()
print("R&D 與 Profit 相關性:", corr.loc['R&D Spend', 'Profit']) # 0.9729`
  },
  3: {
    badge: "Phase 3 • 特徵工程與虛擬變數陷阱防範",
    title: "資料準備 (Data Preparation)",
    desc: "建立無資料洩漏（Data Leakage）的轉換管線，妥善處理類別特徵 State，防止多元迴歸中的虛擬變數陷阱。",
    points: [
      "避免虛擬變數陷阱 (Dummy Variable Trap)：State 有 3 類別，需 drop='first' 僅保留 2 個虛擬變數（Florida、New York），以 California 作為基準對照組。",
      "Scikit-Learn ColumnTransformer：數值變數與類別變數統一以管線化處理，確保上線推論一致性。",
      "可重現資料切分：80% 訓練集 (40 筆)、20% 測試集 (10 筆)，固定 random_state=42。"
    ],
    code: `from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['State']),
        ('num', 'passthrough', ['R&D Spend', 'Administration', 'Marketing'])
    ]
)`
  },
  4: {
    badge: "Phase 4 • 前向特徵選取與多元迴歸建模",
    title: "建立模型與特徵選取 (Modeling & Feature Selection)",
    desc: "透過前向逐步特徵選取（Sequential Forward Selection）找出最佳特徵子集，並比對 Full OLS、Lasso、Ridge 與最優 2-Feature 模型。",
    points: [
      "前向特徵選取 (SFS)：特徵數 k=2 時達到全域最佳，RMSE 降至 $8,198.80，R² 升至 0.9474。",
      "入選最佳特徵：['R&D Spend', 'Marketing Spend']。剔除不具顯著性之行政支出與州別變數。",
      "多元線性迴歸 (Full OLS) vs 精簡模型：驗證奧坎剃刀原則，精簡模型泛化誤差顯著降低約 10.3%。",
      "正則化模型交叉驗證：Lasso (L1) 自動將微弱權重壓縮，與前向選取結論完全吻合。"
    ],
    code: `from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# 前向逐步選取最佳特徵 (k=2)
sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=2, direction='forward')
sfs.fit(X_train, y_train)
print("最佳特徵:", X.columns[sfs.get_support()]) # ['R&D Spend', 'Marketing Spend']`
  },
  5: {
    badge: "Phase 5 • 統計假設檢驗與商業 ROI 結論",
    title: "模型評估 (Evaluation)",
    desc: "依據高斯-馬可夫定理全面檢驗殘差特性，確保迴歸估計量為最佳線性不偏估計量（BLUE）。",
    points: [
      "殘差常態性：Shapiro-Wilk 檢定 W=0.9645, p=0.8362 > 0.05，殘差完全服從常態分佈。",
      "無自相關性：Durbin-Watson 統計量 d=1.821 ≈ 2.0，誤差項相互獨立。",
      "同質變異數：殘差對預測值散佈圖呈均勻水平帶狀分佈，無異質變異數現象。",
      "商業結論：研發投入回報最強 (+$0.81 / $1)；行政投入無顯著獲利拉抬 (-$0.069, p>0.60)；地理位置無顯著利潤差距。"
    ],
    code: `from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import shapiro

y_pred = pipeline.predict(X_test)
print(f"Test R2: {r2_score(y_test, y_pred):.4f}") # 0.8987
print(f"Shapiro p-value: {shapiro(y_test - y_pred).pvalue:.4f}") # 0.8362`
  },
  6: {
    badge: "Phase 6 • 模型序列化、API 封裝與雲端發布",
    title: "成果部署 (Deployment)",
    desc: "將訓練完成的最佳管線序列化儲存為 joblib 檔案，並建構即時預測 CLI 與純前端 Web 應用，支援一鍵發布至 GitHub Pages 或 Vercel。",
    points: [
      "模型持久化：輸出 model/startup_profit_pipeline.pkl 及元數據 model_metadata.json。",
      "推論工具開發：提供 predict.py 支援終端參數推論與 CSV 批次評分。",
      "Web 互動化：本網頁整合即時推論與 50 家新創對比，支援全球免費靜態託管。"
    ],
    code: `import joblib
# 儲存生產管線
joblib.dump(pipeline, 'model/startup_profit_pipeline.pkl')

# 載入並進行單筆推論
loaded_model = joblib.load('model/startup_profit_pipeline.pkl')
sample_pred = loaded_model.predict(new_startup_df)`
  }
};

// ============================================================================
// 3. Preset Scenarios Configuration
// ============================================================================
const PRESETS = {
  deeptech: {
    rd: 165000,
    admin: 120000,
    mkt: 380000,
    state: "California"
  },
  scaleup: {
    rd: 75000,
    admin: 110000,
    mkt: 320000,
    state: "New York"
  },
  bootstrap: {
    rd: 40000,
    admin: 60000,
    mkt: 90000,
    state: "Florida"
  },
  corporate: {
    rd: 25000,
    admin: 155000,
    mkt: 45000,
    state: "California"
  }
};

// ============================================================================
// 4. State & Chart Instances
// ============================================================================
let budgetChartInstance = null;
let impactChartInstance = null;
let currentPhase = 1;

// ============================================================================
// 5. Utility Functions
// ============================================================================
function formatCurrency(num) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(num);
}

function formatNumber(num) {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(num);
}

// ============================================================================
// 6. Multiple Linear Regression Core Calculation
// ============================================================================
function calculateProfit(rd, admin, mkt, state) {
  const { intercept, coef } = MODEL_CONFIG;

  const rdContrib = rd * coef.rd;
  const adminContrib = admin * coef.admin;
  const mktContrib = mkt * coef.marketing;

  let stateContrib = 0;
  if (state === "Florida") {
    stateContrib = coef.florida;
  } else if (state === "New York") {
    stateContrib = coef.new_york;
  } else {
    stateContrib = coef.california; // 0
  }

  const predictedProfit = intercept + rdContrib + adminContrib + mktContrib + stateContrib;
  const totalSpend = rd + admin + mkt;
  const roi = totalSpend > 0 ? (predictedProfit / totalSpend) * 100 : 0;

  return {
    profit: predictedProfit,
    totalSpend: totalSpend,
    roi: roi,
    intercept: intercept,
    rdContrib: rdContrib,
    adminContrib: adminContrib,
    mktContrib: mktContrib,
    stateContrib: stateContrib
  };
}

// ============================================================================
// 7. Update UI & Simulator Displays
// ============================================================================
function updateSimulator() {
  const rd = parseFloat(document.getElementById("rd-slider").value) || 0;
  const admin = parseFloat(document.getElementById("admin-slider").value) || 0;
  const mkt = parseFloat(document.getElementById("mkt-slider").value) || 0;

  const stateChecked = document.querySelector('input[name="state"]:checked');
  const state = stateChecked ? stateChecked.value : "California";

  // Compute ML Prediction
  const result = calculateProfit(rd, admin, mkt, state);

  // Update Display Elements
  document.getElementById("predicted-profit-val").textContent = formatCurrency(result.profit);
  document.getElementById("total-spend-val").textContent = formatCurrency(result.totalSpend);

  const roiEl = document.getElementById("predicted-roi-val");
  roiEl.textContent = (result.roi >= 0 ? "+" : "") + result.roi.toFixed(1) + "%";
  if (result.roi >= 0) {
    roiEl.className = "sub-val text-emerald";
  } else {
    roiEl.className = "sub-val text-rose";
  }

  // Update Linear Attribution Formulas
  document.getElementById("attr-intercept").textContent = "+" + formatCurrency(result.intercept);
  document.getElementById("attr-rd-formula").textContent = `+0.8056 × $${formatNumber(rd)}`;
  document.getElementById("attr-rd-contrib").textContent = (result.rdContrib >= 0 ? "+" : "-") + formatCurrency(Math.abs(result.rdContrib));

  document.getElementById("attr-admin-formula").textContent = `-0.0688 × $${formatNumber(admin)}`;
  document.getElementById("attr-admin-contrib").textContent = (result.adminContrib >= 0 ? "+" : "-") + formatCurrency(Math.abs(result.adminContrib));

  document.getElementById("attr-mkt-formula").textContent = `+0.0299 × $${formatNumber(mkt)}`;
  document.getElementById("attr-mkt-contrib").textContent = (result.mktContrib >= 0 ? "+" : "-") + formatCurrency(Math.abs(result.mktContrib));

  let stateLabel = "California (基準 = $0)";
  if (state === "Florida") stateLabel = "Florida (+$938.79)";
  if (state === "New York") stateLabel = "New York (+$6.99)";
  document.getElementById("attr-state-formula").textContent = stateLabel;
  document.getElementById("attr-state-contrib").textContent = (result.stateContrib >= 0 ? "+" : "-") + formatCurrency(Math.abs(result.stateContrib));

  // Update Charts
  updateCharts(rd, admin, mkt, result);
}

// ============================================================================
// 8. Interactive Charts (Chart.js)
// ============================================================================
function initCharts() {
  const isDark = document.body.getAttribute("data-theme") !== "light";
  const textColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";

  // 1. Budget Share Donut Chart
  const budgetCtx = document.getElementById("budgetDonutChart").getContext("2d");
  budgetChartInstance = new Chart(budgetCtx, {
    type: "doughnut",
    data: {
      labels: ["研發 (R&D)", "行政 (Admin)", "行銷 (Marketing)"],
      datasets: [{
        data: [130000, 115000, 280000],
        backgroundColor: ["#3b82f6", "#f43f5e", "#10b981"],
        borderWidth: 0,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: { color: textColor, font: { size: 10, family: "'Plus Jakarta Sans'" } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: $${formatNumber(ctx.raw)}`
          }
        }
      },
      cutout: "68%"
    }
  });

  // 2. Factor Impact Bar Chart
  const impactCtx = document.getElementById("impactBarChart").getContext("2d");
  impactChartInstance = new Chart(impactCtx, {
    type: "bar",
    data: {
      labels: ["基準截距", "研發貢獻", "行政損耗", "行銷拉抬"],
      datasets: [{
        data: [54028, 104732, -7911, 8360],
        backgroundColor: ["#818cf8", "#10b981", "#f43f5e", "#38bdf8"],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `效益: $${formatNumber(ctx.raw)}`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: textColor, font: { size: 9, family: "'Plus Jakarta Sans'" } },
          grid: { display: false }
        },
        y: {
          ticks: { color: textColor, font: { size: 9, family: "'JetBrains Mono'" } },
          grid: { color: gridColor }
        }
      }
    }
  });
}

function updateCharts(rd, admin, mkt, result) {
  if (budgetChartInstance) {
    budgetChartInstance.data.datasets[0].data = [rd, admin, mkt];
    budgetChartInstance.update("none");
  }

  if (impactChartInstance) {
    impactChartInstance.data.datasets[0].data = [
      Math.round(result.intercept),
      Math.round(result.rdContrib),
      Math.round(result.adminContrib),
      Math.round(result.mktContrib)
    ];
    impactChartInstance.update("none");
  }
}

// ============================================================================
// 9. Input & Slider Synchronizers
// ============================================================================
function setupInputSync(sliderId, inputId) {
  const slider = document.getElementById(sliderId);
  const input = document.getElementById(inputId);

  slider.addEventListener("input", () => {
    input.value = slider.value;
    updateSimulator();
  });

  input.addEventListener("input", () => {
    slider.value = input.value;
    updateSimulator();
  });
}

// ============================================================================
// 10. CRISP-DM Tabs Controller
// ============================================================================
function renderCrispPhase(phaseNum) {
  currentPhase = phaseNum;
  const phaseData = CRISP_PHASES[phaseNum];
  const container = document.getElementById("crisp-card");

  // Update Pill Active state
  document.querySelectorAll(".crisp-pill").forEach(pill => {
    if (parseInt(pill.dataset.phase, 10) === phaseNum) {
      pill.classList.add("active");
    } else {
      pill.classList.remove("active");
    }
  });

  container.innerHTML = `
    <span class="crisp-badge-phase">${phaseData.badge}</span>
    <h3 class="crisp-detail-title">${phaseData.title}</h3>
    <div class="crisp-detail-grid">
      <div class="crisp-text">
        <p>${phaseData.desc}</p>
        <ul class="crisp-keypoints">
          ${phaseData.points.map(pt => `<li>${pt}</li>`).join("")}
        </ul>
      </div>
      <div>
        <div class="crisp-code-preview">${escapeHtml(phaseData.code)}</div>
      </div>
    </div>
  `;
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ============================================================================
// 11. 50 Startups Interactive Table
// ============================================================================
function renderDatasetTable(filterState = "all", searchTerm = "") {
  if (typeof STARTUPS_DATA === "undefined") {
    console.warn("STARTUPS_DATA is not loaded.");
    return;
  }

  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";

  const term = searchTerm.toLowerCase().trim();
  let count = 0;

  STARTUPS_DATA.forEach((row, idx) => {
    const rd = row["R&D Spend"];
    const admin = row["Administration"];
    const mkt = row["Marketing Spend"];
    const state = row["State"];
    const actual = row["Profit"];

    // State filter check
    if (filterState !== "all" && state !== filterState) {
      return;
    }

    // Search filter check
    if (term) {
      const matchState = state.toLowerCase().includes(term);
      const matchNum = `${rd} ${admin} ${mkt} ${actual}`.includes(term);
      if (!matchState && !matchNum) return;
    }

    count++;

    // Compute model prediction & residual error
    const predResult = calculateProfit(rd, admin, mkt, state);
    const pred = predResult.profit;
    const residual = actual - pred;

    let badgeClass = "badge-ca";
    if (state === "New York") badgeClass = "badge-ny";
    if (state === "Florida") badgeClass = "badge-fl";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>#${idx + 1}</td>
      <td class="font-mono">$${formatNumber(rd)}</td>
      <td class="font-mono">$${formatNumber(admin)}</td>
      <td class="font-mono">$${formatNumber(mkt)}</td>
      <td><span class="state-badge ${badgeClass}">${state}</span></td>
      <td class="font-mono font-bold">$${formatNumber(actual)}</td>
      <td class="font-mono text-emerald">$${formatNumber(pred)}</td>
      <td class="font-mono ${residual >= 0 ? 'text-emerald' : 'text-rose'}">
        ${residual >= 0 ? '+' : ''}$${formatNumber(residual)}
      </td>
      <td>
        <button class="btn-table-action" data-idx="${idx}" title="載入至模擬器">試算</button>
      </td>
    `;

    // Click row or button to load
    tr.querySelector(".btn-table-action").addEventListener("click", (e) => {
      e.stopPropagation();
      loadStartupIntoSimulator(row);
    });

    tr.addEventListener("click", () => {
      loadStartupIntoSimulator(row);
    });

    tbody.appendChild(tr);
  });

  document.getElementById("table-record-count").textContent = `顯示 ${count} / ${STARTUPS_DATA.length} 筆紀錄`;
}

function loadStartupIntoSimulator(row) {
  document.getElementById("rd-slider").value = row["R&D Spend"];
  document.getElementById("rd-input").value = row["R&D Spend"];

  document.getElementById("admin-slider").value = row["Administration"];
  document.getElementById("admin-input").value = row["Administration"];

  document.getElementById("mkt-slider").value = row["Marketing Spend"];
  document.getElementById("mkt-input").value = row["Marketing Spend"];

  const stateRadio = document.querySelector(`input[name="state"][value="${row['State']}"]`);
  if (stateRadio) {
    stateRadio.checked = true;
    document.querySelectorAll(".state-option").forEach(opt => {
      if (opt.dataset.state === row['State']) opt.classList.add("active");
      else opt.classList.remove("active");
    });
  }

  // Clear active preset buttons
  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));

  updateSimulator();

  // Scroll to simulator with smooth animation
  document.getElementById("simulator-section").scrollIntoView({ behavior: "smooth" });
}

// ============================================================================
// 12. Theme Toggle Controller
// ============================================================================
function setupThemeToggle() {
  const btn = document.getElementById("theme-toggle-btn");
  const savedTheme = localStorage.getItem("sp_theme") || "dark";
  document.body.setAttribute("data-theme", savedTheme);

  btn.addEventListener("click", () => {
    const current = document.body.getAttribute("data-theme");
    const next = current === "light" ? "dark" : "light";
    document.body.setAttribute("data-theme", next);
    localStorage.setItem("sp_theme", next);

    // Refresh charts colors
    if (budgetChartInstance && impactChartInstance) {
      budgetChartInstance.destroy();
      impactChartInstance.destroy();
      initCharts();
      updateSimulator();
    }
  });
}

// ============================================================================
// 13. Presets & Controls Event Listeners
// ============================================================================
function setupEventListeners() {
  setupInputSync("rd-slider", "rd-input");
  setupInputSync("admin-slider", "admin-input");
  setupInputSync("mkt-slider", "mkt-input");

  // State radio option cards
  document.querySelectorAll(".state-option").forEach(opt => {
    opt.addEventListener("click", () => {
      document.querySelectorAll(".state-option").forEach(o => o.classList.remove("active"));
      opt.classList.add("active");
      const radio = opt.querySelector('input[type="radio"]');
      radio.checked = true;
      updateSimulator();
    });
  });

  // Preset buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const presetKey = btn.dataset.preset;
      const data = PRESETS[presetKey];
      if (data) {
        document.getElementById("rd-slider").value = data.rd;
        document.getElementById("rd-input").value = data.rd;
        document.getElementById("admin-slider").value = data.admin;
        document.getElementById("admin-input").value = data.admin;
        document.getElementById("mkt-slider").value = data.mkt;
        document.getElementById("mkt-input").value = data.mkt;

        const radio = document.querySelector(`input[name="state"][value="${data.state}"]`);
        if (radio) {
          radio.checked = true;
          document.querySelectorAll(".state-option").forEach(o => {
            if (o.dataset.state === data.state) o.classList.add("active");
            else o.classList.remove("active");
          });
        }
        updateSimulator();
      }
    });
  });

  // Reset button
  document.getElementById("btn-reset").addEventListener("click", () => {
    document.getElementById("btn-preset-deeptech").click();
  });

  // Randomize button
  document.getElementById("btn-randomize").addEventListener("click", () => {
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    const rd = Math.round((Math.random() * 180000 + 10000) / 1000) * 1000;
    const admin = Math.round((Math.random() * 140000 + 50000) / 1000) * 1000;
    const mkt = Math.round((Math.random() * 350000 + 10000) / 1000) * 1000;
    const states = ["California", "New York", "Florida"];
    const state = states[Math.floor(Math.random() * states.length)];

    document.getElementById("rd-slider").value = rd;
    document.getElementById("rd-input").value = rd;
    document.getElementById("admin-slider").value = admin;
    document.getElementById("admin-input").value = admin;
    document.getElementById("mkt-slider").value = mkt;
    document.getElementById("mkt-input").value = mkt;

    const radio = document.querySelector(`input[name="state"][value="${state}"]`);
    if (radio) {
      radio.checked = true;
      document.querySelectorAll(".state-option").forEach(o => {
        if (o.dataset.state === state) o.classList.add("active");
        else o.classList.remove("active");
      });
    }
    updateSimulator();
  });

  // CRISP-DM Pill click
  document.querySelectorAll(".crisp-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      const p = parseInt(pill.dataset.phase, 10);
      renderCrispPhase(p);
    });
  });

  // Table filters
  const searchInput = document.getElementById("dataset-search");
  const stateFilter = document.getElementById("state-filter");

  searchInput.addEventListener("input", () => {
    renderDatasetTable(stateFilter.value, searchInput.value);
  });

  stateFilter.addEventListener("change", () => {
    renderDatasetTable(stateFilter.value, searchInput.value);
  });
}

// ============================================================================
// 14. App Initialization
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupThemeToggle();
  initCharts();
  setupEventListeners();
  renderCrispPhase(1);
  renderDatasetTable();
  updateSimulator();
});
