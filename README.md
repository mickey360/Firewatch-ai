# FireWatch AI

A real wildfire-intelligence command center: computer-vision fire/smoke detection, NASA FIRMS satellite observations, Open-Meteo weather context, geospatial risk scoring, image/video analysis, and an explainable directional spread field.

## Stack
Next.js + TypeScript + MapLibre on Vercel; FastAPI + Ultralytics + Hugging Face on a Python service; NASA FIRMS and Open-Meteo for external data.

## Real data / setup
The detector defaults to the real Hugging Face checkpoint `rabahdev/fire-smoke-yolov8n` (`best.pt`). NASA FIRMS requires a real free MAP_KEY; the app never fabricates satellite observations if the key is absent. Open-Meteo requires no API key.

### Web
```bash
cd web && npm install && cp .env.example .env.local && npm run dev
```
Set `NEXT_PUBLIC_API_URL` to the FastAPI URL.

### API
```bash
cd api && python -m venv .venv
# activate it
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
The first detection downloads the real model from Hugging Face and caches it locally.

### Deploy
- `web/` → Vercel.
- `api/` → Render Web Service using `pip install -r requirements.txt` and `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Put the Vercel URL in `CORS_ORIGINS` on the API.
- Put a real NASA FIRMS MAP_KEY in the API environment if satellite observations are required.

Render Free is suitable for portfolio/hobby deployments but is not production emergency infrastructure. FireWatch is decision support, not a certified warning system.
