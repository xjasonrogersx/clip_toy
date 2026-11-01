# CLIP Toy

A local CLIP image and text vector explorer. It uses `openai/clip-vit-base-patch32` by default, stores uploaded images and normalized vectors under `data_cash/`, and searches images with cosine similarity.

## Run

```bash
pip install -r requirements.txt --break-system-packages
python server.py
```

Open http://127.0.0.1:8000/.

Startup loads the cached CLIP weights once before serving the page. This can take several seconds on CPU, but the first text or image request will not pause while loading the model.

The first embedding request downloads the model from Hugging Face and may require network access. Model files are stored in `data_cash/model_cache/` and reused on later runs. Once the local snapshot is complete, the server automatically uses it without Hugging Face metadata requests. Set `CLIP_OFFLINE=1` to force offline mode, or set `CLIP_MODEL` to use another compatible CLIP checkpoint. `CLIP_PORT` changes the port.

## Pages

- `/` uploads images, creates vectors, and visualizes the cached image vectors.
- `/text` embeds a prompt, visualizes its vector, finds similar cached images, and shows `Prompt -> Tokenizer -> CLIP -> Output` in LiteGraph.
- `/graph` shows the image and text branches feeding the shared CLIP embedding space.
- `/api/health` reports model and cache status.

Each upload has an image file, a NumPy vector file, and an entry in `data_cash/index.json`. Set `CLIP_DATA_DIR` to use another storage directory.

## Notes

This is intentionally a small single-process local tool. It does not include authentication, image generation, or a separate vector database. The browser loads LiteGraph from jsDelivr, so graph pages need network access for that script unless it is vendored locally.