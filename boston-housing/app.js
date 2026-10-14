/* BostonPulse AI – interactive browser-side inference */
const BASE = { CRIM:.12, ZN:11, INDUS:11, CHAS:0, NOX:.55, RM:6.3, AGE:68, DIS:4.1, RAD:9, TAX:330, PTRATIO:18.5, B:380, LSTAT:12 };
const FIELDS = [
  ['RM','每戶平均房間數',3,9,.1,'房'], ['PTRATIO','各鎮師生比',12,22,.1,''], ['CRIM','人均犯罪率',0,30,.01,''], ['TAX','每萬美元財產稅率',187,711,1,''], ['NOX','一氧化氮濃度',.38,.87,.01,'']
];
const PRESETS = {
  luxury:{RM:7.5,PTRATIO:15,CRIM:.03,TAX:280,NOX:.42,CHAS:1},
  suburb:{RM:6.3,PTRATIO:18.5,CRIM:.12,TAX:330,NOX:.55,CHAS:0},
  urban:{RM:5.6,PTRATIO:20.5,CRIM:5.5,TAX:666,NOX:.68,CHAS:0},
  budget:{RM:5.1,PTRATIO:21.5,CRIM:15,TAX:666,NOX:.78,CHAS:0}
};
const PHASES = [
  ['業務理解','建立可解釋的房價預測工具，協助理解住宅條件如何影響中位數房價。',['目標變數：MEDV（$1,000）','核心情境：購屋評估與區域比較','成功指標：以 RMSE、R² 與殘差診斷評估']],
  ['資料理解','使用 Kaggle Boston Housing 的 506 筆住宅資料與 13 項結構、環境與社經特徵。',['資料集包含 MEDV 目標欄位','檢查特徵分佈與相關性','RM、LSTAT、PTRATIO 是重要訊號']],
  ['資料準備','以各欄位中位數補齊缺失值，並固定 80/20 訓練與測試切分。',['保留所有 13 項數值特徵','中位數插補避免捨棄樣本','切分 random_state = 42']],
  ['建模','比較完整 OLS 與 Sequential Forward Selection (SFS) 精簡模型。',['Full OLS 使用 13 個特徵','SFS 在 k = 8 取得最低測試 RMSE','互動頁面以模型係數即時計算']],
  ['評估','以預測誤差、殘差圖與 Durbin-Watson 統計量檢視模型品質。',['Durbin-Watson 約 2.26','診斷圖呈現殘差分佈','相關矩陣協助辨識共線性']],
  ['部署','模型係數與 506 筆資料已匯出成靜態 JavaScript，可直接部署至 GitHub Pages。',['不需伺服器即可互動試算','資料表可回填到模擬器','支援明暗主題切換']]
];
let architecture = 'full', impactChart, compareChart;
const money = v => `$${Math.round(v * 1000).toLocaleString('en-US')}`;
const price = v => `${money(v)} (${v.toFixed(1)}K)`;
const values = () => {
  const v = {...BASE}; FIELDS.forEach(([key]) => v[key] = Number(document.getElementById(`${key}-slider`).value));
  v.CHAS = Number(document.querySelector('input[name="chas"]:checked')?.value || 0); return v;
};
function estimate(v, type = architecture) {
  const isOptimal = type === 'optimal', coef = isOptimal ? BOSTON_MODEL.optimalCoefficients : BOSTON_MODEL.coefficients;
  let prediction = isOptimal ? BOSTON_MODEL.optimalIntercept : BOSTON_MODEL.intercept;
  const parts = [['截距', prediction]];
  Object.entries(coef).forEach(([key, coefficient]) => { const amount = (v[key] ?? BASE[key]) * coefficient; prediction += amount; parts.push([key, amount]); });
  return { prediction: Math.max(5, Math.min(50, prediction)), parts };
}
function renderControls() {
  document.getElementById('controls').innerHTML = FIELDS.map(([key,label,min,max,step,unit]) => `<div class="control-group"><div class="control-header"><label class="control-label" for="${key}-input">${label} (${key})</label><input id="${key}-input" class="numeric-input" type="number" min="${min}" max="${max}" step="${step}" value="${BASE[key]}"></div><input id="${key}-slider" class="styled-slider" type="range" min="${min}" max="${max}" step="${step}" value="${BASE[key]}"><div class="slider-ticks"><span>${min}${unit}</span><span>${((Number(min)+Number(max))/2).toFixed(step < 1 ? 1 : 0)}</span><span>${max}${unit}</span></div></div>`).join('') + `<div class="control-group"><label class="control-label">查爾斯河臨河住宅 (CHAS)</label><div class="state-selector"><label class="state-option active"><input name="chas" type="radio" value="0" checked> 非臨河 <span class="state-tag">CHAS = 0</span></label><label class="state-option"><input name="chas" type="radio" value="1"> 臨河 <span class="state-tag">CHAS = 1</span></label></div></div>`;
  FIELDS.forEach(([key]) => { const slider = document.getElementById(`${key}-slider`), input = document.getElementById(`${key}-input`); slider.addEventListener('input', () => { input.value = slider.value; update(); }); input.addEventListener('input', () => { slider.value = input.value; update(); }); });
  document.querySelectorAll('input[name="chas"]').forEach(el => el.addEventListener('change', () => { document.querySelectorAll('.state-option').forEach(x => x.classList.toggle('active', x.querySelector('input').checked)); update(); }));
}
function update() {
  const v = values(), result = estimate(v), full = estimate(v,'full'), optimal = estimate(v,'optimal');
  document.getElementById('predicted-price-val').textContent = money(result.prediction);
  document.getElementById('model-type-label').textContent = architecture === 'full' ? 'Full OLS' : 'SFS Optimal';
  document.getElementById('model-features-count').textContent = architecture === 'full' ? '13 項' : '8 項';
  document.getElementById('predicted-ci-val').textContent = architecture === 'full' ? '± $3,500' : '± $3,300';
  const named = result.parts.filter(([name]) => ['截距','RM','PTRATIO','CRIM','TAX','NOX','CHAS'].includes(name));
  document.getElementById('attr-items-container').innerHTML = named.map(([name, amount]) => `<div class="attr-item"><span class="attr-name">${name === '截距' ? '模型截距' : `${name} · ${BOSTON_MODEL.featureDescriptions[name] || ''}`}<small>${name === '截距' ? 'baseline' : `貢獻值`}</small></span><b class="attr-contrib ${amount >= 0 ? 'positive':'negative'}">${amount >= 0 ? '+' : '−'}${money(Math.abs(amount))}</b></div>`).join('');
  if (impactChart) { impactChart.data.labels = named.map(x => x[0]); impactChart.data.datasets[0].data = named.map(x => +(x[1]*1000).toFixed(0)); impactChart.update('none'); compareChart.data.datasets[0].data = [+(full.prediction*1000).toFixed(0), +(optimal.prediction*1000).toFixed(0)]; compareChart.update('none'); }
}
function initCharts() {
  const color = document.body.dataset.theme === 'light' ? '#475569' : '#94a3b8';
  impactChart = new Chart(document.getElementById('featureImpactChart'), {type:'bar',data:{labels:[],datasets:[{data:[],backgroundColor:['#818cf8','#34d399','#fb7185','#38bdf8','#f59e0b','#a78bfa','#22c55e'],borderRadius:5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>money(c.raw/1000)}}},scales:{x:{ticks:{color,font:{size:9}},grid:{display:false}},y:{ticks:{color,callback:x=>'$'+(x/1000)+'K'},grid:{color:'rgba(148,163,184,.12)'}}}}});
  compareChart = new Chart(document.getElementById('modelCompareChart'), {type:'bar',data:{labels:['Full OLS','SFS Optimal'],datasets:[{data:[],backgroundColor:['#818cf8','#34d399'],borderRadius:7}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>money(c.raw/1000)}}},scales:{x:{ticks:{color},grid:{display:false}},y:{ticks:{color,callback:x=>'$'+(x/1000)+'K'},grid:{color:'rgba(148,163,184,.12)'}}}}});
}
function renderPhases(active=0) { document.getElementById('crisp-pills').innerHTML = PHASES.map(([title],i)=>`<button class="crisp-pill ${i===active?'active':''}" data-phase="${i}"><span class="pill-num">${i+1}</span><span class="pill-text">${title}</span></button>`).join(''); const [title,desc,points] = PHASES[active]; document.getElementById('crisp-card').innerHTML = `<span class="crisp-badge-phase">Phase ${active+1} · CRISP-DM</span><h3 class="crisp-detail-title">${title}</h3><div class="crisp-detail-grid"><div class="crisp-text"><p>${desc}</p><ul class="crisp-keypoints">${points.map(x=>`<li>${x}</li>`).join('')}</ul></div><pre class="crisp-code-preview">${active===3 ? "Sequential Forward Selection\n最佳子集：RM, PTRATIO, TAX, NOX\nCRIM, B, INDUS, DIS" : "Boston Housing Dataset\n506 samples · 13 features\nScikit-Learn LinearRegression"}</pre></div>`; document.querySelectorAll('.crisp-pill').forEach(btn => btn.onclick=()=>renderPhases(Number(btn.dataset.phase))); }
function renderTable() { const term=document.getElementById('dataset-search').value.trim().toLowerCase(), filter=document.getElementById('medv-filter').value, rows=BOSTON_DATA.filter(row=>{const m=row.MEDV; return (filter==='all'||filter==='low'&&m<15||filter==='mid'&&m>=15&&m<=30||filter==='high'&&m>30) && (!term||Object.values(row).join(' ').toLowerCase().includes(term));}); document.getElementById('table-body').innerHTML = rows.map(row=>{const pred=estimate(row,'full').prediction, err=row.MEDV-pred; return `<tr data-index="${BOSTON_DATA.indexOf(row)}"><td>#${BOSTON_DATA.indexOf(row)+1}</td><td>${row.RM.toFixed(2)}</td><td>${row.CRIM.toFixed(2)}</td><td>${row.PTRATIO.toFixed(1)}</td><td>${row.TAX}</td><td>${row.NOX.toFixed(2)}</td><td>${row.LSTAT.toFixed(1)}</td><td>${row.CHAS?'臨河':'否'}</td><td class="font-mono font-bold">${price(row.MEDV)}</td><td class="font-mono text-emerald">${price(pred)}</td><td class="font-mono ${err>=0?'text-emerald':'text-rose'}">${err>=0?'+':''}${money(err)}</td><td><button class="btn-table-action">帶入</button></td></tr>`;}).join(''); document.getElementById('table-record-count').textContent=`顯示 ${rows.length} / ${BOSTON_DATA.length} 筆紀錄`; document.querySelectorAll('#table-body tr').forEach(tr=>tr.onclick=()=>loadRow(BOSTON_DATA[tr.dataset.index])); }
function loadRow(row) { FIELDS.forEach(([key])=>{document.getElementById(`${key}-slider`).value=row[key];document.getElementById(`${key}-input`).value=row[key];}); const radio=document.querySelector(`input[name="chas"][value="${row.CHAS}"]`); if(radio) radio.checked=true; document.querySelectorAll('.state-option').forEach(x=>x.classList.toggle('active',x.querySelector('input').checked)); update(); document.getElementById('simulator-section').scrollIntoView({behavior:'smooth'}); }
document.addEventListener('DOMContentLoaded',()=>{renderControls();initCharts();renderPhases();renderTable();update(); document.querySelectorAll('.preset-btn').forEach(btn=>btn.onclick=()=>{const p=PRESETS[btn.dataset.preset];loadRow({...BASE,...p});document.querySelectorAll('.preset-btn').forEach(x=>x.classList.toggle('active',x===btn));}); document.querySelectorAll('.model-arch-btn').forEach(btn=>btn.onclick=()=>{architecture=btn.dataset.arch;document.querySelectorAll('.model-arch-btn').forEach(x=>x.classList.toggle('active',x===btn));update();}); document.getElementById('btn-reset').onclick=()=>loadRow(BASE); document.getElementById('btn-randomize').onclick=()=>{const row=BOSTON_DATA[Math.floor(Math.random()*BOSTON_DATA.length)];loadRow(row);}; document.getElementById('dataset-search').oninput=renderTable;document.getElementById('medv-filter').onchange=renderTable;document.getElementById('theme-toggle-btn').onclick=()=>{document.body.dataset.theme=document.body.dataset.theme==='dark'?'light':'dark';document.getElementById('theme-toggle-btn').textContent=document.body.dataset.theme==='dark'?'☼':'☾';};});
