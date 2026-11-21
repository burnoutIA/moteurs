// telecharger-poids.mjs <dossier> : le checkpoint officiel d'Inkling 2 Mini (NVFP4 + MXFP8), avec reprise.
import fs from 'node:fs'; import path from 'node:path';
const dir = process.argv[2], base = 'https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/';
const files = ['config.json', 'generation_config.json', 'hf_quant_config.json', 'model.safetensors.index.json', 'tokenizer.json',
  'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'processor_config.json', 'tiktoken/tokenizer.model',
  'mtp.safetensors', 'model-00001-of-00001.safetensors'];
for (const f of files) {
  const dest = path.join(dir, f); fs.mkdirSync(path.dirname(dest), { recursive: true });
  const head = await fetch(base + f, { method: 'HEAD', redirect: 'follow' }); const size = Number(head.headers.get('content-length'));
  let have = fs.existsSync(dest) ? fs.statSync(dest).size : 0;
  if (size && have === size) { console.log('déjà là', f); continue; }
  const r = await fetch(base + f, { headers: have ? { Range: `bytes=${have}-` } : {}, redirect: 'follow' });
  if (r.status !== 206) have = 0;
  const w = fs.createWriteStream(dest, { flags: have ? 'a' : 'w' }); let got = have, last = 0;
  for await (const c of r.body) { if (!w.write(c)) await new Promise(o => w.once('drain', o)); got += c.length;
    if (got - last > 1 << 30) { last = got; console.log(f, (got / 2 ** 30).toFixed(1), '/', (size / 2 ** 30).toFixed(1), 'Gio'); } }
  await new Promise(o => w.end(o)); console.log('fini', f, !size || got === size ? 'ok' : `TAILLE ${got} ≠ ${size}`);
}
