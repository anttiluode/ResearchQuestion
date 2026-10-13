/* Independent dashboard for the measured, learned v2 results.
 * No browser model is trained here. Data come from the committed fixed-seed receipt.
 */
(() => {
  const world = document.getElementById('learned-world');
  const metric = document.getElementById('learned-metric');
  const status = document.getElementById('learned-status');
  if (!world || !metric || !status) return;
  let data = null;
  const methods = ['predictive4', 'predictive5', 'pca5', 'gru5'];
  function render() {
    if (!data) return;
    const key = metric.value;
    const choice = data.means[world.value];
    if (!choice) return;
    const values = methods.map(name => choice[name][key]);
    const maximum = Math.max(.3, ...values) * 1.04;
    methods.forEach((name, i) => {
      const number = document.querySelector('[data-method-value="' + name + '"]');
      const bar = document.querySelector('[data-method-bar="' + name + '"]');
      if (number) number.textContent = values[i].toFixed(3);
      if (bar) {
        bar.style.width = Math.max(1.5, 100 * values[i] / maximum).toFixed(2) + '%';
        bar.setAttribute('aria-label', name + ': ' + values[i].toFixed(3) + ' normalized RMSE');
      }
    });
    status.textContent = 'Two independent apparatus seeds · 400 held-out episodes per seed · ' +
      (key === 'future_nrmse' ? 'six-step open-loop' : key === 'early_nrmse' ? 'early steps 1–3' : 'late steps 4–6');
  }
  world.addEventListener('change', render);
  metric.addEventListener('change', render);
  fetch('./results/learned_v2_summary.json')
    .then(response => {
      if (!response.ok) throw new Error('Result data not accessible');
      return response.json();
    })
    .then(result => {
      if (!result.means || !result.means.bridge || !result.means.closed) {
        throw new Error('Unexpected receipt format');
      }
      data = result;
      render();
    })
    .catch(() => {
      world.disabled = true;
      metric.disabled = true;
      status.textContent = 'Showing committed bridge-world summary; dynamic result data could not load.';
    });
})();
