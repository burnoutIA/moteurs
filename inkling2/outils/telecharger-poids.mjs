// telecharger-poids.mjs <dossier> : le checkpoint officiel d'Inkling 2 Mini (NVFP4 + MXFP8), avec reprise, et
// l'encodeur d'image au format de llama.cpp (mmproj-BF16.gguf, Unsloth), pour build/inkling2-vision.
import fs from 'node:fs'; import path from 'node:path';
const dir = process.argv[2], official = 'https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/';
const unsloth = 'https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/';
const files = ['config.json', 'generation_config.json', 'hf_quant_config.json', 'model.safetensors.index.json', 'tokenizer.json',
  'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'processor_config.json', 'tiktoken/tokenizer.model',
  'mtp.safetensors', 'model-00001-of-00001.safetensors', 'mmproj-BF16.gguf'];
for (const f of files) {
  const base = f.startsWith('mmproj') ? unsloth : official;
  const dest = path.join(dir, f); fs.mkdirSync(path.dirname(dest), { recursive: true });
  const head = await fetch(base + f, { method: 'HEAD', redirect: 'follow' }); const size = Number(head.headers.get('content-length'));
  let have = fs.existsSync(dest) ? fs.statSync(dest).size : 0;
  if (size && have === size) { console.log('déjà là', f); continue; }
  // Les petits fichiers hors LFS arrivent en gzip, sans taille : pas de reprise, on les reprend en entier (sinon
  // la reprise demande la suite d'un fichier complet, reçoit un 416 et écrit son texte à la place du fichier).
  if (!size || have > size) have = 0;
  const r = await fetch(base + f, { headers: have ? { Range: `bytes=${have}-` } : {}, redirect: 'follow' });
  if (r.status !== 200 && r.status !== 206) throw new Error(`${f} : HTTP ${r.status}`);
  if (r.status !== 206) have = 0;
  const w = fs.createWriteStream(dest, { flags: have ? 'a' : 'w' }); let got = have, last = 0;
  for await (const c of r.body) { if (!w.write(c)) await new Promise(o => w.once('drain', o)); got += c.length;
    if (got - last > 1 << 30) { last = got; console.log(f, (got / 2 ** 30).toFixed(1), '/', (size / 2 ** 30).toFixed(1), 'Gio'); } }
  await new Promise(o => w.end(o)); console.log('fini', f, !size || got === size ? 'ok' : `TAILLE ${got} ≠ ${size}`);
}
