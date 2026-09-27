/* Diagnostic card for the experimental MPC response learner. No dependencies. */
class MpcResponseCard extends HTMLElement {
  setConfig(config) {
    if (!config.entity) throw new Error('Set entity to the Better Thermostat climate entity');
    this.config = config;
    if (!this.shadowRoot) this.attachShadow({mode: 'open'});
    this.render();
  }
  set hass(hass) { this._hass = hass; this.render(); }
  getCardSize() { return 8; }
  render() {
    if (!this.config || !this._hass) return;
    const state = this._hass.states[this.config.entity];
    const d = state?.attributes?.mpc_v2_response_curve;
    const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const title = esc(this.config.title || 'Learned valve heating response');
    const status = esc(d?.status?.replaceAll('_', ' ') || 'Learning disabled or no diagnostics yet');
    const finite = n => typeof n === 'number' && Number.isFinite(n);
    const x = p => 65 + p * 5.15;
    const ys = [...(d?.heat_K_min || []), ...(d?.upper_K_min || []), ...(d?.control_heat_K_min || [])].filter(finite);
    const maximum = Math.max(.04, ...ys) * 1.15;
    const y = v => 235 - 175 * v / maximum;
    const points = (d?.opening_pct || []).map((p,i) => ({p, m:d.heat_K_min[i], lo:d.lower_K_min[i], hi:d.upper_K_min[i], n:d.independent_holds[i]}));
    const episodes = d?.version >= 2;
    const total = episodes ? (d.accepted_episodes || 0) : points.reduce((n,p) => n+p.n,0);
    const unit = episodes ? 'independent episodes' : 'independent holds';
    const age = finite(d?.report_age_min) ? `${Math.round(d.report_age_min)} min` : 'unknown';
    const computed = finite(d?.computed_at) ? new Date(d.computed_at * 1000).toLocaleString() : '';
    const pending = episodes ? `<p>${d.pending_reports || 0} real reports in the current episode · ${Math.round(d.pending_duration_min || 0)} min · latest report ${age} old</p><p class="muted">Last evaluated: ${esc(computed)}. Heartbeat allowance: ${Math.round(d.heartbeat_limit_min || 90)} min.</p>` : '';
    const countMax = Math.max(6, ...points.map(p=>p.n));
    let shapes = '';
    for (let i=0;i<=4;i++) {
      const v=maximum*i/4;
      shapes += `<line x1="65" x2="580" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="57" y="${y(v)+4}" text-anchor="end">${v.toFixed(3)}</text>`;
    }
    for (let p=0;p<=100;p+=20) shapes += `<text x="${x(p)}" y="253" text-anchor="middle">${p}%</text><text x="${x(p)}" y="388" text-anchor="middle">${p}%</text>`;
    const control = d?.control_heat_K_min;
    if (control) shapes += `<polyline points="${control.map((v,i)=>`${x(d.opening_pct[i])},${y(v)}`).join(' ')}" fill="none" stroke="#9aa5b4" stroke-width="2" stroke-dasharray="5 4"/>`;
    for (const p of points) {
      if (finite(p.lo) && finite(p.hi)) {
        shapes += `<rect x="${x(p.p)-6}" y="${y(p.hi)}" width="12" height="${y(p.lo)-y(p.hi)}" fill="#369ac8" opacity=".2"/><path d="M${x(p.p)},${y(p.lo)}V${y(p.hi)} M${x(p.p)-5},${y(p.lo)}h10 M${x(p.p)-5},${y(p.hi)}h10" stroke="#369ac8" fill="none"/>`;
      }
      if (finite(p.m) && p.n) shapes += `<circle cx="${x(p.p)}" cy="${y(p.m)}" r="4" fill="#369ac8"><title>${p.p}%: ${p.m.toFixed(4)} K/min, ${p.n} ${unit}</title></circle>`;
      if (p.n) shapes += `<rect x="${x(p.p)-6}" y="${365-50*p.n/countMax}" width="12" height="${50*p.n/countMax}" fill="#369ac8"/><text x="${x(p.p)}" y="${359-50*p.n/countMax}" text-anchor="middle">${p.n}</text>`;
    }
    const cap = d?.saturation_pct;
    if (finite(cap)) shapes += `<line x1="${x(cap)}" x2="${x(cap)}" y1="55" y2="235" stroke="#d99a31" stroke-dasharray="3 3"/>`;
    this.shadowRoot.innerHTML = `<style>
      :host {display:block;} ha-card {display:block; box-sizing:border-box; border-radius:12px; padding:20px; color:var(--primary-text-color,#dbe5f1); background:var(--ha-card-background,var(--card-background-color,#18202c));}
      h2 {font-size:19px; margin:0 0 8px;} p {font-size:13px; line-height:1.5; margin:8px 0;} .muted {color:var(--secondary-text-color,#a8b2c2);}
      svg {width:100%; height:auto; font:12px system-ui; color:inherit;} text {fill:currentColor;} .grid {stroke:currentColor; opacity:.12;} strong {font-weight:650;}
      .pill {display:inline-block; padding:4px 8px; border:1px solid #627188; border-radius:5px; margin-right:8px;}
    </style><ha-card><h2>${title}</h2>
      <p><span class="pill">${episodes ? 'Observation only; existing MPC controls' : (d?.control_active ? 'Learned control active' : 'Prior control; collecting evidence')}</span>${status}</p>
      <svg viewBox="0 0 620 430" role="img" aria-label="Estimated heating response, uncertainty and independent episode coverage">
        <text x="65" y="30">Delivered room heating · K/min</text>${shapes}
        ${total ? '' : '<text x="320" y="135" text-anchor="middle">No accepted heating evidence yet</text>'}
        <text x="65" y="285">${episodes ? 'Episode coverage by opening' : 'Independent holds per opening'}</text><text x="320" y="415" text-anchor="middle">Valve opening command</text>
      </svg>
      <p><strong>Estimated effective saturation: ${finite(cap) ? `${cap}%` : 'unknown'}</strong> · ${total} ${unit}</p>${pending}
      <p class="muted">${episodes ? 'Blue: fitted candidate response. Bands show model sensitivity and differences between episodes, not 95% confidence intervals. Empty ranges have no supported estimate. Candidates do not affect heating.' : 'Blue: observed median and pointwise 95% confidence intervals (at least six holds). Dashed: control curve including assumptions. Empty positions are unobserved.'} This is heating response, not measured water flow.</p>
      <p class="muted">${episodes ? 'No saturation claim is made during observation. Normal sensor silence adds no measurements; collection waits for a real report.' : 'Saturation means additional observed opening adds at most about 15% of full-opening heat within the uncertainty bounds.'} The configured cap is unchanged.</p>
    </ha-card>`;
  }
}
if (!customElements.get('mpc-response-card')) customElements.define('mpc-response-card', MpcResponseCard);
window.customCards = window.customCards || [];
window.customCards.push({type:'mpc-response-card',name:'MPC response diagnostics',description:'Learned heating response, uncertainty and coverage'});
