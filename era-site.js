/* v2b classical-check panel. Reads the committed ERA summary; trains nothing. */
(() => {
  const world = document.getElementById('era-world');
  const status = document.getElementById('era-status');
  if (!world || !status) return;
  const rows = ['predictive5', 'era5', 'era5_pem_posthoc'];
  let data = null;
  function value(name) {
    if (name === 'predictive5') return data.learned_v2_reference[world.value].predictive5;
    return data.means[world.value][name].future_nrmse;
  }
  function render() {
    if (!data) return;
    const values = rows.map(value);
    const maximum = Math.max(.3, ...values) * 1.04;
    rows.forEach((name, i) => {
      const number = document.querySelector('[data-era-value="' + name + '"]');
      const bar = document.querySelector('[data-era-bar="' + name + '"]');
      if (number) number.textContent = values[i].toFixed(3);
      if (bar) {
        bar.style.width = Math.max(1.5, 100 * values[i] / maximum).toFixed(2) + '%';
        bar.setAttribute('aria-label', name + ': ' + values[i].toFixed(3) + ' normalized RMSE');
      }
    });
    status.textContent = 'Mean of apparatus seeds 41 and 97 · 400 held-out episodes per seed · ' +
      (world.value === 'bridge' ? 'bridge feeds back' : 'bridge cannot feed back');
  }
  world.addEventListener('change', render);
  fetch('./results/era_v2b_summary.json')
    .then(response => {
      if (!response.ok) throw new Error('Result data not accessible');
      return response.json();
    })
    .then(result => {
      if (!result.means || !result.learned_v2_reference) throw new Error('Unexpected format');
      data = result;
      render();
    })
    .catch(() => {
      world.disabled = true;
      status.textContent = 'Showing committed bridge-world summary; dynamic result data could not load.';
    });
})();
