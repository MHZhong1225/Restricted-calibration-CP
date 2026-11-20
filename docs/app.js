// Table 1 of the supplied SGCP manuscript. Each metric is [mean, standard deviation].
const methods = ['Marginal', 'Partial', 'Exhaustive', 'RLCP', 'CluCP', 'AFCP', 'FaReG', 'SGCP'];
const benchmarks = {
  SYN: {
    description: 'Synthetic multiclass diagnosis · 6 classes · intersectional calibration heterogeneity',
    rows: [
      [[89.52,1.52],[2.309,.078],[14.5,1.6]],
      [[94.50,1.12],[2.419,.048],[6.2,.7]],
      [[95.85,.91],[3.342,.053],[6.0,.6]],
      [[93.10,2.45],[2.540,.305],[4.5,1.8]],
      [[89.80,1.78],[2.426,.050],[4.9,.8]],
      [[91.77,1.23],[2.432,.043],[4.6,.5]],
      [[92.61,.98],[2.579,.029],[4.7,.5]],
      [[90.13,.39],[2.225,.035],[4.2,.7]]
    ]
  },
  Nursery: {
    description: 'Nursery admission decisions · 4 classes · controlled subgroup imbalance and label noise',
    rows: [
      [[84.15,5.20],[2.170,.260],[20.4,5.2]],
      [[88.97,3.80],[1.850,.200],[12.1,2.5]],
      [[96.80,3.10],[2.850,.220],[15.5,3.2]],
      [[88.67,6.20],[1.220,.410],[13.5,3.8]],
      [[85.08,1.60],[1.121,.064],[12.8,1.3]],
      [[92.85,.84],[1.085,.024],[14.1,.6]],
      [[93.21,.62],[1.230,.012],[13.3,.6]],
      [[91.65,.70],[1.089,.035],[11.8,.7]]
    ]
  },
  'MIMIC-IV': {
    description: 'Clinical tabular benchmark · race, gender, and insurance subgroups · main-table results',
    rows: [
      [[90.79,10.62],[1.226,.087],[27.1,7.1]],
      [[94.50,4.54],[1.258,.140],[8.0,3.0]],
      [[96.45,4.22],[2.800,.142],[19.7,3.2]],
      [[92.50,2.80],[1.350,.100],[1.7,.8]],
      [[92.22,1.28],[1.302,.059],[3.5,.8]],
      null,
      [[90.23,.09],[1.174,.001],[.5,.0]],
      [[90.13,.07],[1.107,.001],[.8,.0]]
    ]
  },
  BACH: {
    description: 'Breast histopathology · 4 classes · class labels serve as the subgroup variable',
    rows: [
      [[89.20,9.64],[2.894,.267],[15.7,1.7]],
      [[96.57,8.08],[3.385,.299],[7.6,1.6]],
      [[97.41,6.89],[3.867,.118],[11.6,1.5]],
      [[91.88,10.56],[2.311,.319],[8.1,2.5]],
      [[95.91,3.59],[3.257,.220],[10.7,2.6]],
      [[95.00,2.60],[2.830,.237],[6.3,1.7]],
      [[93.60,3.40],[2.959,.239],[7.4,1.5]],
      [[91.90,2.70],[2.224,.215],[6.2,1.1]]
    ]
  }
};

function renderBenchmark(name) {
  const {description, rows} = benchmarks[name];
  const available = rows.filter(Boolean);
  const minimum = [null, Math.min(...available.map(row => row[1][0])), Math.min(...available.map(row => row[2][0]))];
  const digits = [2, 3, 1];
  document.querySelectorAll('[data-dataset]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.dataset === name));
  });
  document.querySelector('#dataset-description').textContent = description;
  document.querySelector('#table-caption').textContent = `${name} · mean ± standard deviation over 20 runs`;
  document.querySelector('#results-body').innerHTML = rows.map((row, index) => {
    const cells = row ? row.map(([mean, sd], metric) => {
      const value = mean.toFixed(digits[metric]);
      return `<td>${mean === minimum[metric] ? `<strong>${value}</strong>` : value}<span class="sd"> ± ${sd.toFixed(digits[metric])}</span></td>`;
    }).join('') : '<td colspan="3">Not reported</td>';
    return `<tr class="${methods[index] === 'SGCP' ? 'ours' : ''}"><th scope="row">${methods[index]}</th>${cells}</tr>`;
  }).join('');
  const sgcp = rows[7];
  document.querySelector('#result-summary').innerHTML = [
    [sgcp[0][0].toFixed(2) + '%', 'SGCP coverage'],
    [sgcp[1][0].toFixed(3), 'SGCP average set size'],
    [sgcp[2][0].toFixed(1) + ' pp', 'SGCP coverage gap']
  ].map(([value, label]) => `<div><strong>${value}</strong><span>${label}</span></div>`).join('');
  const maxGap = Math.max(...available.map(row => row[2][0]));
  document.querySelector('#gap-chart').innerHTML = rows.map((row, index) => {
    if (!row) return '';
    const gap = row[2][0];
    return `<div class="bar-row ${methods[index] === 'SGCP' ? 'ours' : ''}"><span>${methods[index]}</span><div class="bar-track" aria-hidden="true"><div class="bar" style="width:${100 * gap / maxGap}%"></div></div><span class="bar-value">${gap.toFixed(1)}</span></div>`;
  }).join('');
}

document.querySelectorAll('[data-dataset]').forEach(button => {
  button.addEventListener('click', () => renderBenchmark(button.dataset.dataset));
});
document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C (or ⌘C) to copy.';
  }
});
renderBenchmark('SYN');
