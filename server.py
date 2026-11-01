from __future__ import annotations

import cgi
import io
import json
import logging
import mimetypes
import os
import threading
from time import perf_counter
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / os.environ.get("CLIP_DATA_DIR", "data_cash")
IMAGE_DIR = DATA_DIR / "images"
VECTOR_DIR = DATA_DIR / "vectors"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"
INDEX_PATH = DATA_DIR / "index.json"
MODEL_NAME = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
OFFLINE_MODE = os.environ.get("CLIP_OFFLINE", "0").lower() in {"1", "true", "yes"}
HOST = os.environ.get("CLIP_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLIP_PORT", "8000"))
LOGGER = logging.getLogger("clip_toy")


def _model_cache_complete() -> bool:
	repository = MODEL_CACHE_DIR / f"models--{MODEL_NAME.replace('/', '--')}" / "snapshots"
	if not repository.is_dir():
		return False
	for snapshot in repository.iterdir():
		if not snapshot.is_dir():
			continue
		configuration = snapshot / "config.json"
		processor = snapshot / "preprocessor_config.json"
		weights = snapshot / "pytorch_model.bin"
		if configuration.is_file() and processor.is_file() and weights.is_file() and weights.stat().st_size > 0:
			return True
	return False


LOCAL_ONLY = OFFLINE_MODE or _model_cache_complete()


def _json_bytes(value: object) -> bytes:
	return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _vector_summary(vector: np.ndarray) -> dict[str, object]:
	values = vector.astype(np.float32).reshape(-1)
	return {
		"dimensions": int(values.size),
		"min": round(float(values.min()), 6),
		"max": round(float(values.max()), 6),
		"mean": round(float(values.mean()), 6),
		"std": round(float(values.std()), 6),
		"sample": [round(float(value), 6) for value in values[:64]],
	}


class VectorStore:
	def __init__(self) -> None:
		for directory in (IMAGE_DIR, VECTOR_DIR):
			directory.mkdir(parents=True, exist_ok=True)
		self.lock = threading.RLock()
		self.items: dict[str, dict[str, object]] = {}
		self._load()

	def _load(self) -> None:
		if INDEX_PATH.is_file():
			try:
				loaded = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
				if isinstance(loaded, dict):
					self.items = {str(key): value for key, value in loaded.items() if isinstance(value, dict)}
			except (OSError, json.JSONDecodeError):
				LOGGER.warning("Ignoring unreadable vector index at %s", INDEX_PATH)

	def _save(self) -> None:
		temporary = INDEX_PATH.with_suffix(".tmp")
		temporary.write_text(json.dumps(self.items, indent=2), encoding="utf-8")
		temporary.replace(INDEX_PATH)

	def add(self, image_bytes: bytes, filename: str, vector: np.ndarray) -> dict[str, object]:
		image_id = uuid.uuid4().hex
		suffix = Path(filename).suffix.lower()
		if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
			suffix = ".png"
		image_path = IMAGE_DIR / f"{image_id}{suffix}"
		vector_path = VECTOR_DIR / f"{image_id}.npy"
		image_path.write_bytes(image_bytes)
		np.save(vector_path, vector.astype(np.float32))
		item = {"id": image_id, "filename": Path(filename).name, "image": image_path.name, "vector": vector_path.name}
		with self.lock:
			self.items[image_id] = item
			self._save()
		return item

	def list(self) -> list[dict[str, object]]:
		with self.lock:
			return list(self.items.values())

	def get(self, image_id: str) -> dict[str, object] | None:
		with self.lock:
			return self.items.get(image_id)

	def vector(self, image_id: str) -> np.ndarray | None:
		with self.lock:
			item = self.items.get(image_id)
		if item is None:
			return None
		try:
			return np.load(VECTOR_DIR / str(item["vector"])).astype(np.float32).reshape(-1)
		except (OSError, ValueError, KeyError):
			return None

	def delete(self, image_id: str) -> bool:
		with self.lock:
			item = self.items.pop(image_id, None)
			if item is None:
				return False
			for directory, key in ((IMAGE_DIR, "image"), (VECTOR_DIR, "vector")):
				filename = Path(str(item[key])).name
				(directory / filename).unlink(missing_ok=True)
			self._save()
			return True

	def search(self, vector: np.ndarray, limit: int = 12) -> list[dict[str, object]]:
		results: list[dict[str, object]] = []
		with self.lock:
			items = list(self.items.values())
		for item in items:
			try:
				stored = np.load(VECTOR_DIR / str(item["vector"])).astype(np.float32).reshape(-1)
				score = float(np.dot(vector, stored))
			except (OSError, ValueError, KeyError):
				continue
			result = dict(item)
			result["score"] = round(score, 6)
			results.append(result)
		results.sort(key=lambda result: float(result["score"]), reverse=True)
		return results[:limit]


class ClipEngine:
	def __init__(self) -> None:
		self.lock = threading.Lock()
		self._processor = None
		self._model = None
		self.error: str | None = None

	def _load(self):
		if self._model is None:
			try:
				load_started = perf_counter()
				LOGGER.info("CLIP load started: model=%s local_only=%s cache=%s", MODEL_NAME, LOCAL_ONLY, MODEL_CACHE_DIR)
				if LOCAL_ONLY:
					os.environ["HF_HUB_OFFLINE"] = "1"
					os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
				from transformers import CLIPModel, CLIPProcessor
				MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
				load_options = {
					"cache_dir": str(MODEL_CACHE_DIR),
					"local_files_only": LOCAL_ONLY,
				}
				self._processor = CLIPProcessor.from_pretrained(MODEL_NAME, **load_options)
				self._model = CLIPModel.from_pretrained(MODEL_NAME, **load_options)
				self._model.eval()
				LOGGER.info("CLIP load finished in %.2fs", perf_counter() - load_started)
			except Exception as error:
				self.error = str(error)
				LOGGER.exception("CLIP load failed")
				raise RuntimeError(f"Unable to load {MODEL_NAME}: {error}") from error
		return self._processor, self._model

	@staticmethod
	def _normalise(vector) -> np.ndarray:
		values = vector.detach().cpu().numpy().astype(np.float32).reshape(-1)
		norm = float(np.linalg.norm(values))
		return values / norm if norm else values

	def image(self, image: Image.Image) -> np.ndarray:
		with self.lock:
			processor, model = self._load()
			inputs = processor(images=image.convert("RGB"), return_tensors="pt")
			with __import__("torch").inference_mode():
				vision_output = model.vision_model(pixel_values=inputs["pixel_values"])
				features = model.visual_projection(vision_output.pooler_output)
				return self._normalise(features)

	def text(self, prompt: str, padding_tokens: int = 0) -> tuple[np.ndarray, tuple[int, ...], int]:
		with self.lock:
			started = perf_counter()
			padding_tokens = max(0, min(77, int(padding_tokens)))
			LOGGER.info("Text embedding started: characters=%d padding_tokens=%d", len(prompt), padding_tokens)
			processor, model = self._load()
			base_inputs = processor.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=77)
			prompt_token_count = int(base_inputs["attention_mask"].sum().item())
			target_length = min(77, max(1, int(base_inputs["input_ids"].shape[1]) + padding_tokens))
			inputs = processor.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=target_length)
			LOGGER.info("Text tokenization finished: prompt_tokens=%d input_shape=%s", prompt_token_count, tuple(inputs["input_ids"].shape))
			inference_started = perf_counter()
			with __import__("torch").inference_mode():
				text_output = model.text_model(
					input_ids=inputs["input_ids"],
					attention_mask=inputs.get("attention_mask"),
				)
				features = model.text_projection(text_output.pooler_output)
				result = self._normalise(features)
			inference_ms = (perf_counter() - inference_started) * 1000
			LOGGER.info("Text embedding finished in %.2fs: dimensions=%d", perf_counter() - started, result.size)
			return result, tuple(inputs["input_ids"].shape), prompt_token_count, inference_ms


STORE = VectorStore()
ENGINE = ClipEngine()


def _page(title: str, body: str, script: str = "") -> str:
	return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/litegraph.js@0.7.18/build/litegraph.css"><style>
:root{{--ink:#17212b;--muted:#607080;--line:#cbd8df;--paper:#fbfaf6;--teal:#0d6870;--orange:#c75d2d;--blue:#2266a8}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:16px/1.45 Georgia,serif;background:radial-gradient(circle at 8% 0,#f4cba955,transparent 30%),linear-gradient(135deg,#edf5f2,#f8f2e8)}}main{{max-width:1180px;margin:auto;padding:28px 18px 60px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:22px}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1;margin:0 0 10px;letter-spacing:0}}h2{{margin-top:0}}p{{color:var(--muted)}}nav{{display:flex;gap:14px;flex-wrap:wrap}}a{{color:var(--teal);font-weight:bold}}.panel{{background:#ffffffd9;border:1px solid #ffffff;border-radius:12px;padding:20px;box-shadow:0 14px 40px #17354c18;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}label{{display:block;font-weight:bold;margin:10px 0 6px}}input[type=text],textarea{{width:100%;padding:12px;border:1px solid var(--line);border-radius:7px;font:inherit;background:var(--paper)}}textarea{{min-height:110px;resize:vertical}}button{{border:0;border-radius:7px;padding:11px 16px;background:var(--orange);color:white;font:inherit;font-weight:bold;cursor:pointer}}button:disabled{{opacity:.55}}.status{{min-height:24px;color:var(--muted);margin:10px 0}}.error{{color:#a52c2c}}.preview{{max-width:100%;max-height:300px;display:block;border-radius:8px;margin-top:12px}}.vector{{height:150px;display:flex;align-items:center;gap:2px;overflow:hidden;padding:10px;background:#102a3840;border-radius:7px}}.bar{{flex:1;min-width:2px;background:var(--teal);border-radius:2px 2px 0 0}}.bar.negative{{background:var(--orange);align-self:flex-end}}.meta{{font-family:ui-monospace,monospace;font-size:.85rem;color:var(--muted);white-space:pre-wrap}}.results{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}}.result{{border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff}}.result img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:5px}}.graph{{height:430px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#dce8e7}}.wide{{grid-column:1/-1}}@media(max-width:700px){{header{{display:block}}nav{{margin-top:16px}}main{{padding:20px 12px}}}}
</style></head><body><main>{body}</main><script src="https://cdn.jsdelivr.net/npm/litegraph.js@0.7.18/build/litegraph.js"></script><script>{script}</script></body></html>'''


COMMON_NAV = '<nav><a href="/">Images</a><a href="/text">Text search</a><a href="/graph">LiteGraph</a></nav>'


def image_page() -> str:
	body = f'''<header><div><h1>Image vectors</h1><p>Upload an image, encode it with CLIP, and keep the vector in the local cache.</p></div>{COMMON_NAV}</header><section class="panel"><form id="image-form"><label for="image-file">Image</label><input id="image-file" type="file" accept="image/png,image/jpeg,image/webp" required><button>Generate image vector</button></form><div id="image-status" class="status">Ready · {len(STORE.list())} cached images</div><img id="image-preview" class="preview" hidden><div id="image-meta" class="meta"></div><div id="image-vector" class="vector" aria-label="Image vector visualization"></div></section><section class="panel"><h2>Cached images</h2><div id="cached" class="results"></div></section>'''
	script = '''const $=id=>document.getElementById(id);const status=$('image-status');function drawInto(box,vector){box.replaceChildren();const sample=vector.sample||[];const max=Math.max(...sample.map(v=>Math.abs(v)),1e-6);sample.forEach(v=>{const bar=document.createElement('i');bar.className='bar'+(v<0?' negative':'');bar.style.height=`${Math.max(3,Math.abs(v)/max*100)}%`;bar.title=v;box.append(bar)})}function draw(vector){drawInto($('image-vector'),vector)}async function loadMini(item,card){try{const r=await fetch(`/api/images/${item.id}/vector`),vector=await r.json();if(r.ok){drawInto(card.querySelector('[data-mini-vector]'),vector);card.querySelector('[data-mini-meta]').textContent=`${vector.dimensions} dimensions · mean ${vector.mean}`}}catch(x){card.querySelector('[data-mini-meta]').textContent='Vector unavailable'}}async function selectImage(item){status.textContent=`Loading vector for ${item.filename}…`;try{const r=await fetch(`/api/images/${item.id}/vector`),vector=await r.json();if(!r.ok)throw Error(vector.error);$('image-preview').src=`/api/images/${item.id}`;$('image-preview').hidden=false;$('image-meta').textContent=JSON.stringify(vector,null,2);draw(vector);status.textContent=`Selected ${item.filename}`;document.querySelectorAll('.result').forEach(card=>card.classList.toggle('selected',card.dataset.id===item.id))}catch(x){status.textContent=x.message;status.className='status error'}}function render(items){$('cached').innerHTML=items.map(x=>`<article class="result" style="grid-column:1/-1;display:grid;grid-template-columns:minmax(140px,220px) 1fr;gap:16px;align-items:start" data-id="${x.id}"><div><button class="image-select" style="padding:0;background:transparent;width:100%;margin-top:0" type="button" data-select="${x.id}"><img src="/api/images/${x.id}" alt="${x.filename}"></button><button class="delete" style="background:#9e3f36" type="button" data-delete="${x.id}">Delete</button></div><div><b>${x.filename}</b><div class="vector" data-mini-vector aria-label="Vector visualization for ${x.filename}"></div><span class="meta" data-mini-meta>Loading vector…</span></div></article>`).join('')||'<p>No images cached yet.</p>';items.forEach(item=>{const card=document.querySelector(`.result[data-id="${item.id}"]`);card.querySelector('[data-select]').addEventListener('click',()=>selectImage(item));card.querySelector('[data-delete]').addEventListener('click',async()=>{if(!confirm(`Delete ${item.filename}?`))return;const r=await fetch(`/api/images/${item.id}`,{method:'DELETE'});const result=await r.json();if(!r.ok){status.textContent=result.error;status.className='status error';return}if($('image-preview').src.endsWith(item.id)){$('image-preview').hidden=true;$('image-meta').textContent='';$('image-vector').replaceChildren()}status.textContent=`Deleted ${item.filename}`;refresh()});loadMini(item,card)})}async function refresh(){const r=await fetch('/api/images');render(await r.json())}$('image-form').addEventListener('submit',async e=>{e.preventDefault();const file=$('image-file').files[0];if(!file)return;status.textContent='Encoding image…';const data=new FormData();data.append('image',file);try{const r=await fetch('/api/images/embed',{method:'POST',body:data}),j=await r.json();if(!r.ok)throw Error(j.error);await selectImage(j.image);status.textContent=`Saved ${j.image.id}`;refresh()}catch(x){status.textContent=x.message;status.className='status error'}});refresh();'''
	body = body.replace('id="image-meta"', 'id="image-meta" hidden')
	return _page("Image vectors", body, script)


def text_page() -> str:
	body = f'''<header><div><h1>Text search</h1><p>Encode a prompt and find the cached images with the closest CLIP meaning.</p></div>{COMMON_NAV}</header><section class="panel"><label for="prompt">Prompt</label><textarea id="prompt" placeholder="a quiet red house beside the sea"></textarea><label for="padding">Additional token padding <output id="padding-value">0</output></label><input id="padding" type="range" min="0" max="77" value="0" step="1"><button id="run">Generate text vector</button><div id="status" class="status">Ready · image and text vectors share one CLIP space</div><div id="meta" class="meta"></div><div id="vector" class="vector"></div></section><section class="panel"><div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap"><h2 style="margin:0">Prompt pipeline</h2><button id="save-text-graph" type="button">Save PNG</button></div><div id="text-graph" class="graph"></div></section><section class="panel"><h2>Most similar images</h2><div id="results" class="results"><p>Run a prompt to search cached images.</p></div></section>'''
	script = '''const $=id=>document.getElementById(id);function draw(vector){const box=$('vector');box.replaceChildren();const a=vector.sample||[],m=Math.max(...a.map(v=>Math.abs(v)),1e-6);a.forEach(v=>{const e=document.createElement('i');e.className='bar'+(v<0?' negative':'');e.style.height=`${Math.max(3,Math.abs(v)/m*100)}%`;e.title=v;box.append(e)})}function results(items){$('results').innerHTML=items.map(x=>`<article class="result"><img src="/api/images/${x.id}"><b>${Number(x.score).toFixed(4)}</b><br><span class="meta">${x.filename}</span></article>`).join('')||'<p>No cached images match yet.</p>'}const graph=new LiteGraph.LGraph(),canvas=document.createElement('canvas');canvas.width=1100;canvas.height=400;$('text-graph').append(canvas);const gc=new LiteGraph.LGraphCanvas(canvas,graph);function Stage(){this.properties={}}LiteGraph.registerNodeType('clip/stage',Stage);function node(title,x,y,color){const n=LiteGraph.createNode('clip/stage');n.title=title;n.pos=[x,y];n.size=[220,150];n.color=color;n.bgcolor='#18333f';return n}const prompt=node('Prompt',40,120,'#e28b54');prompt.addOutput('text','string');prompt.addWidget('text','Value','Waiting',v=>{prompt.properties.value=v;prompt.setOutputData(0,v)});const tokenizer=node('Tokenizer',330,120,'#4fc5bd');tokenizer.addInput('text','string');tokenizer.addOutput('tokens','tensor');tokenizer.addWidget('text','Status','Waiting',()=>{},{read_only:true});const clip=node('CLIP encoder',620,120,'#9b8de1');clip.addInput('tokens','tensor');clip.addOutput('embedding','vector');clip.addWidget('text','Status','Waiting',()=>{},{read_only:true});const output=node('Output vector',910,120,'#e28b54');output.addInput('embedding','vector');output.addWidget('text','Dimensions','—',()=>{},{read_only:true});[prompt,tokenizer,clip,output].forEach(n=>graph.add(n));prompt.connect(0,tokenizer,0);tokenizer.connect(0,clip,0);clip.connect(0,output,0);function set(n,k,v){const w=n.widgets.find(x=>x.name===k);if(w)w.value=String(v)}function fit(){gc.ds.scale=.72;gc.ds.offset=[20,20];gc.setDirty(true,true)}fit();$('run').onclick=async()=>{const value=$('prompt').value.trim();if(!value){$('status').textContent='Enter a prompt first.';return}set(prompt,'Value',value);set(tokenizer,'Status','Tokenizing…');set(clip,'Status','Encoding…');$('status').textContent='Encoding prompt…';try{const r=await fetch('/api/text/embed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:value})}),j=await r.json();if(!r.ok)throw Error(j.error);set(tokenizer,'Status',`${j.tokens} tokens`);set(clip,'Status','Complete');set(output,'Dimensions',j.vector.dimensions);$('meta').textContent=JSON.stringify(j.vector,null,2);draw(j.vector);results(j.matches);$('status').textContent=`Found ${j.matches.length} cached image(s).`}catch(x){$('status').textContent=x.message;$('status').className='status error';set(clip,'Status','Failed')}};'''
	script = script.replace("prompt.addWidget('text','Value','Waiting',v=>{prompt.properties.value=v;prompt.setOutputData(0,v)});", "prompt.addWidget('text','Value','Waiting',v=>{prompt.properties.value=v;prompt.setOutputData(0,v)});prompt.addWidget('text','Prompt text','Waiting',()=>{}, {read_only:true});")
	script = script.replace("tokenizer.addWidget('text','Status','Waiting',()=>{},{read_only:true});", "tokenizer.addWidget('text','Status','Waiting',()=>{},{read_only:true});tokenizer.addWidget('text','Token count','—',()=>{},{read_only:true});tokenizer.addWidget('text','Input tensor','—',()=>{},{read_only:true});")
	script = script.replace("clip.addWidget('text','Status','Waiting',()=>{},{read_only:true});", "clip.addWidget('text','Status','Waiting',()=>{},{read_only:true});clip.addWidget('text','Inference time','—',()=>{},{read_only:true});clip.addWidget('text','Input tensor','—',()=>{},{read_only:true});clip.addWidget('text','Output vector','—',()=>{},{read_only:true});")
	script = script.replace("output.addWidget('text','Dimensions','—',()=>{},{read_only:true});", "output.addWidget('text','Dimensions','—',()=>{},{read_only:true});output.addWidget('text','Vector tensor','—',()=>{},{read_only:true});")
	script = script.replace("set(tokenizer,'Status',`${j.tokens} tokens`);set(clip,'Status','Complete');set(output,'Dimensions',j.vector.dimensions);", "set(prompt,'Prompt text',value);set(tokenizer,'Status',`${j.tokens} tokens · +${j.padding_tokens} padding`);set(tokenizer,'Token count',j.tokens);set(tokenizer,'Input tensor',`[${j.input_shape.join(' × ')}]`);set(clip,'Status','Complete');set(clip,'Inference time',`${j.inference_ms.toFixed(2)} ms`);set(clip,'Input tensor',`[${j.input_shape.join(' × ')}]`);set(clip,'Output vector',`[${j.output_shape.join(' × ')}]`);set(output,'Dimensions',j.vector.dimensions);set(output,'Vector tensor',`[${j.output_shape.join(' × ')}]`);")
	script = script.replace("$('run').onclick=async()=>{", "$('padding').addEventListener('input',event=>{$('padding-value').textContent=event.target.value});$('run').onclick=async()=>{")
	script = script.replace("body:JSON.stringify({prompt:value})", "body:JSON.stringify({prompt:value,padding_tokens:Number($('padding').value)})")
	script += '''const similarPanel=$('results').closest('.panel');const pipelinePanel=$('text-graph').closest('.panel');similarPanel.after(pipelinePanel);const similarList=$('results');function formatSimilar(){const cards=[...similarList.querySelectorAll('.result')];cards.sort((a,b)=>Number(b.querySelector('b')?.textContent||-Infinity)-Number(a.querySelector('b')?.textContent||-Infinity)).forEach(card=>similarList.append(card));cards.forEach(card=>{if(card.dataset.formatted)return;const image=card.querySelector('img');if(!image)return;const imageId=image.src.split('/').pop();const score=Number(card.querySelector('b')?.textContent||0);const filename=card.querySelector('.meta')?.textContent||'';card.dataset.formatted='true';card.style.cssText='grid-column:1/-1;display:grid;grid-template-columns:minmax(140px,220px) 1fr;gap:16px;align-items:start';card.innerHTML=`<div><div style="font-weight:bold;font-size:1.15rem">Confidence<br>${score.toFixed(4)}</div><img src="/api/images/${imageId}" alt="${filename}"></div><div><b>${filename}</b><div class="vector" data-similar-vector aria-label="Vector visualization for ${filename}"></div><span class="meta" data-similar-meta>Loading vector…</span></div>`;fetch(`/api/images/${imageId}/vector`).then(response=>response.json()).then(vector=>{if(!vector.dimensions)return;drawIntoSimilar(card.querySelector('[data-similar-vector]'),vector);card.querySelector('[data-similar-meta]').textContent=`${vector.dimensions} dimensions · mean ${vector.mean}`}).catch(()=>{card.querySelector('[data-similar-meta]').textContent='Vector unavailable'})})}function drawIntoSimilar(box,vector){box.replaceChildren();const sample=vector.sample||[];const max=Math.max(...sample.map(value=>Math.abs(value)),1e-6);sample.forEach(value=>{const bar=document.createElement('i');bar.className='bar'+(value<0?' negative':'');bar.style.height=`${Math.max(3,Math.abs(value)/max*100)}%`;bar.title=value;box.append(bar)})}new MutationObserver(formatSimilar).observe(similarList,{childList:true});formatSimilar();'''
	script = script.replace('new MutationObserver(formatSimilar).observe(similarList,{childList:true});formatSimilar();', 'const renderSimilarResults=results;results=(items)=>{renderSimilarResults(items);formatSimilar()};formatSimilar();')
	script = script.replace("similarPanel.after(pipelinePanel);", "")
	script += '''function saveTextGraph(){const button=$('save-text-graph');button.disabled=true;button.textContent='Rendering…';let exporter=null;try{const nodes=graph._nodes;if(!nodes.length)throw Error('There are no graph nodes to export.');const padding=48,topPadding=112,minX=Math.min(...nodes.map(node=>node.pos[0])),minY=Math.min(...nodes.map(node=>node.pos[1])),maxX=Math.max(...nodes.map(node=>node.pos[0]+node.size[0])),maxY=Math.max(...nodes.map(node=>node.pos[1]+node.size[1])),scale=Math.min(2,12000/(maxX-minX+padding*2),12000/(maxY-minY+topPadding+padding)),exportCanvas=document.createElement('canvas');exportCanvas.width=Math.ceil((maxX-minX)*scale+padding*2);exportCanvas.height=Math.ceil((maxY-minY)*scale+topPadding+padding);exporter=new LiteGraph.LGraphCanvas(exportCanvas,graph,{skip_render:true,skip_events:true});exporter.background_image=null;exporter.clear_background=true;exporter.clear_background_color=null;exporter.show_info=false;exporter.render_canvas_border=false;exporter.node_title_color='#000000';exporter.ds.scale=scale;exporter.ds.offset=[padding/scale-minX,topPadding/scale-minY];exporter.onRenderBackground=()=>true;exporter.draw(true,true);exportCanvas.toBlob(blob=>{if(!blob)throw Error('The browser could not create the PNG export.');const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='clip-text-graph.png';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);button.textContent='PNG saved'},'image/png')}catch(error){window.alert(error.message);button.textContent='Save PNG'}finally{if(exporter)graph.detachCanvas(exporter);setTimeout(()=>{button.disabled=false;button.textContent='Save PNG'},1200)}}$('save-text-graph').addEventListener('click',saveTextGraph);'''
	body = body.replace('id="meta"', 'id="meta" hidden')
	return _page("Text search", body, script)


def graph_page() -> str:
	body = f'''<header><div><h1>CLIP graph</h1><p>Inspect the two CLIP modes: image and text inputs share the same embedding space.</p></div>{COMMON_NAV}</header><section class="panel"><div style="display:flex;justify-content:flex-end;margin-bottom:12px"><button id="save-clip-graph" type="button">Save PNG</button></div><div id="graph" class="graph"></div></section>'''
	script = '''const canvas=document.createElement('canvas');canvas.width=1100;canvas.height=430;document.getElementById('graph').append(canvas);const g=new LiteGraph.LGraph(),c=new LiteGraph.LGraphCanvas(canvas,g);function Stage(){this.properties={}}if(!LiteGraph.registered_node_types['clip/stage'])LiteGraph.registerNodeType('clip/stage',Stage);function N(t,x,y,col){const n=LiteGraph.createNode('clip/stage');if(!n)throw new Error('Unable to create CLIP graph stage node');n.title=t;n.pos=[x,y];n.size=[230,150];n.color=col;n.bgcolor='#18333f';return n}function add(n){g.add(n);return n}const image=add(N('Image input',40,50,'#e28b54'));image.addOutput('pixels','image');image.addWidget('text','Mode','Upload image',()=>{},{read_only:true});const text=add(N('Text input',40,250,'#e28b54'));text.addOutput('prompt','string');text.addWidget('text','Mode','Prompt',()=>{},{read_only:true});const tokenizer=add(N('Tokenizer',340,250,'#4fc5bd'));tokenizer.addInput('prompt','string');tokenizer.addOutput('tokens','tensor');const encoder=add(N('Shared CLIP',640,145,'#9b8de1'));encoder.addInput('image','image');encoder.addInput('tokens','tensor');encoder.addOutput('vector','vector');encoder.addWidget('text','Status','Ready',()=>{},{read_only:true});const output=add(N('Vector output',940,145,'#e28b54'));output.addInput('vector','vector');image.connect(0,encoder,0);text.connect(0,tokenizer,0);tokenizer.connect(0,encoder,1);encoder.connect(0,output,0);c.ds.scale=.72;c.ds.offset=[20,20];c.setDirty(true,true);'''
	script += '''function saveClipGraph(){const button=document.getElementById('save-clip-graph');button.disabled=true;button.textContent='Rendering…';let exporter=null;try{const nodes=g._nodes;if(!nodes.length)throw Error('There are no graph nodes to export.');const padding=64,topPadding=112,minX=Math.min(...nodes.map(node=>node.pos[0])),minY=Math.min(...nodes.map(node=>node.pos[1])),maxX=Math.max(...nodes.map(node=>node.pos[0]+node.size[0])),maxY=Math.max(...nodes.map(node=>node.pos[1]+node.size[1])),scale=Math.min(2,14000/(maxX-minX+padding*2),14000/(maxY-minY+topPadding+padding)),exportCanvas=document.createElement('canvas');exportCanvas.width=Math.ceil((maxX-minX)*scale+padding*2);exportCanvas.height=Math.ceil((maxY-minY)*scale+topPadding+padding);exporter=new LiteGraph.LGraphCanvas(exportCanvas,g,{skip_render:true,skip_events:true});exporter.background_image=null;exporter.clear_background=true;exporter.clear_background_color=null;exporter.show_info=false;exporter.render_canvas_border=false;exporter.node_title_color='#000000';exporter.ds.scale=scale;exporter.ds.offset=[padding/scale-minX,topPadding/scale-minY];exporter.onRenderBackground=()=>true;exporter.draw(true,true);exportCanvas.toBlob(blob=>{if(!blob)throw Error('The browser could not create the PNG export.');const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='clip-graph.png';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);button.textContent='PNG saved'},'image/png')}catch(error){window.alert(error.message);button.textContent='Save PNG'}finally{if(exporter)g.detachCanvas(exporter);setTimeout(()=>{button.disabled=false;button.textContent='Save PNG'},1200)}}document.getElementById('save-clip-graph').addEventListener('click',saveClipGraph);'''
	return _page("CLIP graph", body, script)


def _send(handler: BaseHTTPRequestHandler, status: HTTPStatus, body: bytes, content_type: str = "application/json") -> None:
	handler.send_response(status);handler.send_header("Content-Type", content_type);handler.send_header("Content-Length", str(len(body)));handler.end_headers();handler.wfile.write(body)


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, value: object) -> None:
	_send(handler, status, _json_bytes(value))


class Handler(BaseHTTPRequestHandler):
	def log_message(self, format: str, *args: object) -> None:
		LOGGER.info("%s - %s", self.address_string(), format % args)

	def do_GET(self) -> None:
		path = urlparse(self.path).path
		if path == "/": return _send(self, HTTPStatus.OK, image_page().encode(), "text/html; charset=utf-8")
		if path == "/text": return _send(self, HTTPStatus.OK, text_page().encode(), "text/html; charset=utf-8")
		if path == "/graph": return _send(self, HTTPStatus.OK, graph_page().encode(), "text/html; charset=utf-8")
		if path == "/api/health": return _json_response(self, HTTPStatus.OK, {"model": MODEL_NAME, "model_cache": str(MODEL_CACHE_DIR), "offline": LOCAL_ONLY, "cached_images": len(STORE.list()), "model_loaded": ENGINE._model is not None, "model_error": ENGINE.error})
		if path == "/api/images": return _json_response(self, HTTPStatus.OK, STORE.list())
		if path.startswith("/api/images/") and path.endswith("/vector"):
			image_id = unquote(path.removeprefix("/api/images/").removesuffix("/vector"))
			vector = STORE.vector(image_id)
			if vector is None:
				return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "vector not found"})
			return _json_response(self, HTTPStatus.OK, _vector_summary(vector))
		if path.startswith("/api/images/"):
			image_id = unquote(path.removeprefix("/api/images/"))
			item = STORE.get(image_id)
			if item is None: return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "image not found"})
			image_path = IMAGE_DIR / str(item["image"])
			if not image_path.is_file(): return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "image file not found"})
			return _send(self, HTTPStatus.OK, image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "application/octet-stream")
		return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

	def do_DELETE(self) -> None:
		path = urlparse(self.path).path
		if path.startswith("/api/images/"):
			image_id = unquote(path.removeprefix("/api/images/"))
			if STORE.delete(image_id):
				return _json_response(self, HTTPStatus.OK, {"deleted": image_id})
			return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "image not found"})
		_json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

	def do_POST(self) -> None:
		path = urlparse(self.path).path
		try:
			if path == "/api/images/embed":
				content_type = self.headers.get("Content-Type", "")
				if "multipart/form-data" not in content_type: raise ValueError("Expected multipart image upload.")
				form = cgi.FieldStorage(
					fp=self.rfile,
					headers=self.headers,
					environ={
						"REQUEST_METHOD": "POST",
						"CONTENT_TYPE": content_type,
						"CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
					},
				)
				field = form["image"] if "image" in form else None
				if field is None or not getattr(field, "file", None): raise ValueError("Select an image first.")
				raw = field.file.read()
				image = Image.open(io.BytesIO(raw)).convert("RGB")
				vector = ENGINE.image(image)
				item = STORE.add(raw, getattr(field, "filename", "upload.png") or "upload.png", vector)
				return _json_response(self, HTTPStatus.OK, {"image": item, "vector": _vector_summary(vector)})
			if path == "/api/text/embed":
				request_started = perf_counter()
				LOGGER.info("POST /api/text/embed started")
				length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length))
				prompt = payload.get("prompt") if isinstance(payload, dict) else None
				if not isinstance(prompt, str) or not prompt.strip(): raise ValueError("Prompt must not be empty.")
				padding_tokens = payload.get("padding_tokens", 0) if isinstance(payload, dict) else 0
				if isinstance(padding_tokens, bool) or not isinstance(padding_tokens, int) or not 0 <= padding_tokens <= 77:
					raise ValueError("padding_tokens must be an integer from 0 to 77.")
				vector, input_shape, prompt_token_count, inference_ms = ENGINE.text(prompt.strip(), padding_tokens)
				matches = STORE.search(vector)
				LOGGER.info("Text similarity search finished: matches=%d total=%.2fs", len(matches), perf_counter() - request_started)
				return _json_response(self, HTTPStatus.OK, {"vector": _vector_summary(vector), "tokens": prompt_token_count, "padding_tokens": padding_tokens, "input_shape": list(input_shape), "output_shape": [1, int(vector.size)], "inference_ms": round(inference_ms, 2), "matches": matches})
			raise FileNotFoundError
		except ValueError as error:
			LOGGER.warning("Bad request %s %s: %s", self.command, path, error)
			_json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(error)})
		except (OSError, RuntimeError, json.JSONDecodeError) as error:
			LOGGER.exception("Request failed: %s %s", self.command, path)
			_json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	try:
		LOGGER.info("Loading CLIP model from %s", MODEL_CACHE_DIR)
		ENGINE._load()
		LOGGER.info("CLIP model ready")
	except RuntimeError as error:
		LOGGER.warning("CLIP model will load on first embedding request: %s", error)
	server = ThreadingHTTPServer((HOST, PORT), Handler)
	LOGGER.info("CLIP explorer listening at http://%s:%s", HOST, PORT)
	server.serve_forever()


if __name__ == "__main__":
	main()
