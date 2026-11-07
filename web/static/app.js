(() => {
  const form = document.getElementById('forge');
  const def = document.getElementById('definition');
  const pos = document.getElementById('pos');
  const go = document.getElementById('go');
  const out = document.getElementById('result');
  const err = document.getElementById('error');
  const loading = document.getElementById('loading');

  const sentence = (s) => {
    s = s.trim();
    s = s.charAt(0).toUpperCase() + s.slice(1);
    return /[.!?]$/.test(s) ? s : s + '.';
  };

  const show = (data) => {
    const entry = document.createElement('article');
    entry.className = 'entry';

    const head = document.createElement('div');
    head.className = 'entry-head';
    const word = document.createElement('h2');
    word.className = 'headword';
    word.textContent = data.word;
    const p = document.createElement('span');
    p.className = 'pos';
    p.textContent = data.pos;
    head.append(word, p);

    const sense = document.createElement('p');
    sense.className = 'sense';
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = '1.';
    const text = document.createElement('span');
    text.textContent = sentence(data.definition);
    sense.append(n, text);

    entry.append(head, sense);
    out.replaceChildren(entry);
  };

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const definition = def.value.trim();
    err.hidden = true;
    if (!definition) {
      err.textContent = 'Enter a definition first.';
      err.hidden = false;
      def.focus();
      return;
    }
    go.disabled = true;
    out.replaceChildren();
    const timer = setTimeout(() => { loading.hidden = false; }, 120);   // no flicker on fast responses
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ definition, pos: pos.value }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Something went wrong. Please try again.');
      show(data);
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
    } finally {
      clearTimeout(timer);
      loading.hidden = true;
      go.disabled = false;
    }
  });

  def.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) form.requestSubmit();
  });
})();
