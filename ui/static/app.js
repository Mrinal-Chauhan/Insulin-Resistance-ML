// ── Navbar active link on scroll ─────────────────────────────────────────────
const sections = document.querySelectorAll('section[id], .hero[id]');
const navLinks  = document.querySelectorAll('.nav-link');

const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(l => l.classList.remove('active'));
      const link = document.querySelector(`.nav-link[href="#${e.target.id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });

sections.forEach(s => observer.observe(s));

// ── Load meta (dropdowns) ────────────────────────────────────────────────────
async function loadMeta() {
  try {
    const res = await fetch('/api/meta');
    const data = await res.json();

    populate('f_country',    data.countries);
    populate('f_gender',     data.genders);
    populate('f_age_group',  data.age_groups);
    populate('f_heart_risk', data.heart_risks);
  } catch (e) {
    console.error('Failed to load meta:', e);
  }
}

function populate(id, values) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = '';
  values.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
}

// ── Chart tabs ───────────────────────────────────────────────────────────────
const chartLoaded = {};

function activateChartTab(tab) {
  document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.chart-panel').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  const key = tab.dataset.tab;
  document.getElementById('tab-' + key).classList.add('active');
  if (!chartLoaded[key]) loadChart(key);
}

document.querySelectorAll('.chart-tab').forEach(t => {
  t.addEventListener('click', () => activateChartTab(t));
});

const chartEndpoints = {
  'class':    '/api/charts/class-distribution',
  'features': '/api/charts/feature-distributions',
  'corr':     '/api/charts/correlation',
  'cm':       '/api/charts/confusion-matrix',
};

async function loadChart(key) {
  const frame = document.getElementById('chart-' + key);
  if (!frame) return;
  frame.innerHTML = '<div class="chart-loading">Loading chart...</div>';
  try {
    const res = await fetch(chartEndpoints[key]);
    const data = await res.json();
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + data.image;
    img.alt = key + ' chart';
    img.addEventListener('click', () => openLightbox(img.src));
    frame.innerHTML = '';
    frame.appendChild(img);
    chartLoaded[key] = true;
  } catch (e) {
    frame.innerHTML = '<div class="chart-loading">Error loading chart.</div>';
  }
}

// Load the first chart on page load
loadChart('class');

// ── Code tabs ────────────────────────────────────────────────────────────────
document.querySelectorAll('.code-tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.code-tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.code-panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('ctab-' + t.dataset.ctab).classList.add('active');
  });
});

// ── Lightbox ─────────────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lbImg').src = src;
  document.getElementById('lightbox').classList.add('open');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}

document.getElementById('lbClose').addEventListener('click', closeLightbox);
document.getElementById('lbOverlay').addEventListener('click', closeLightbox);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Prediction ────────────────────────────────────────────────────────────────
document.getElementById('predictForm').addEventListener('submit', async e => {
  e.preventDefault();

  const btn = document.getElementById('submitBtn');
  btn.textContent = 'Running...';
  btn.classList.add('loading');

  const form = e.target;
  const payload = {
    country:    form.country.value,
    year:       parseInt(form.year.value),
    age:        parseInt(form.age.value),
    gender:     form.gender.value,
    age_group:  form.age_group.value,
    hba1c:      parseFloat(form.hba1c.value),
    hdl:        parseFloat(form.hdl.value),
    ldl:        parseFloat(form.ldl.value),
    tg:         parseFloat(form.tg.value),
    heart_risk: form.heart_risk.value,
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await res.json();
    renderResult(data, payload);
  } catch (err) {
    renderError(err.message);
  } finally {
    btn.textContent = 'Run Prediction';
    btn.classList.remove('loading');
  }
});

function renderResult(data, inputs) {
  const panel    = document.getElementById('resultPanel');
  const isIR     = data.prediction === 1;
  const prob     = data.probability;
  const probHigh = prob > 50;

  panel.innerHTML = `
    <div class="result-output">
      <div class="result-label-tag">PREDICTION RESULT</div>
      <div class="result-verdict ${isIR ? 'positive' : 'negative'}">
        ${data.label}
      </div>
      <div class="result-pred-code">prediction = ${data.prediction}</div>

      <div class="result-prob-bar-wrap">
        <div class="result-prob-label">
          <span>IR Probability</span>
          <span>${prob}%</span>
        </div>
        <div class="result-prob-track">
          <div class="result-prob-fill ${probHigh ? 'high' : ''}" style="width: ${prob}%"></div>
        </div>
      </div>

      <div class="result-details">
        <div class="result-detail-row">
          <span class="detail-key">Age</span>
          <span class="detail-val">${inputs.age}</span>
        </div>
        <div class="result-detail-row">
          <span class="detail-key">Gender</span>
          <span class="detail-val">${inputs.gender}</span>
        </div>
        <div class="result-detail-row">
          <span class="detail-key">HbA1c</span>
          <span class="detail-val">${inputs.hba1c}</span>
        </div>
        <div class="result-detail-row">
          <span class="detail-key">HDL / LDL / TG</span>
          <span class="detail-val">${inputs.hdl} / ${inputs.ldl} / ${inputs.tg}</span>
        </div>
        <div class="result-detail-row">
          <span class="detail-key">Heart Risk</span>
          <span class="detail-val">${inputs.heart_risk}</span>
        </div>
        <div class="result-detail-row">
          <span class="detail-key">Country</span>
          <span class="detail-val">${inputs.country}</span>
        </div>
      </div>
    </div>
  `;
}

function renderError(msg) {
  const panel = document.getElementById('resultPanel');
  panel.innerHTML = `
    <div style="text-align:center; width: 100%;">
      <div style="font-size: 32px; color: var(--warn); margin-bottom: 12px;">✕</div>
      <div style="font-size: 13px; color: var(--warn); font-weight: 600; margin-bottom: 8px;">Prediction Failed</div>
      <div style="font-size: 12px; color: var(--text-muted); font-family: var(--font-sans);">${msg}</div>
    </div>
  `;
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadMeta();
