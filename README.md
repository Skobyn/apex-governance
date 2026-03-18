# Watson Governance Center

Real-time LLM governance dashboard for the OpenClaw agent ecosystem.

## Features

- **Real-time agent network graph** — nodes pulse when calls fire
- **Live call feed** — scrolling ticker of every agent call
- **Hallucination heatmap** — 24hr rolling view by agent and hour
- **Agent DNA fingerprinting** — 8-dim semantic fingerprint per agent, drift detection
- **Cost burn console** — real-time $ counter with burn rate gauge
- **Decision provenance tree** — click any call for full scoring breakdown
- **WebSocket live updates** — no page refresh needed

## Architecture

```
FastAPI backend
├── POST /api/ingest     — receive agent events
├── GET  /api/stats      — global stats + heatmap
├── GET  /api/agents     — per-agent health
├── GET  /api/calls      — recent call records
├── GET  /api/drift      — DNA fingerprints
├── GET  /api/costs      — cost breakdown
└── WS   /api/live       — real-time event stream
```

## OpenClaw Integration

```bash
# ingest.sh usage
./ingest.sh \
  --agent watson \
  --model claude-sonnet-4-6 \
  --tokens-in 1500 \
  --tokens-out 400 \
  --latency-ms 2100 \
  --output "response text"
```

## Local Dev

```bash
pip install -r requirements.txt
cd app && uvicorn main:app --reload --port 8080
```

## Deploy

Push to `main` — GitHub Actions builds and deploys to Cloud Run.

## Design

- Background: `#0a0f1a`
- Accent: `#00f5a0` neon green
- Panels: `#1B2A4A`
- Font: Inter + JetBrains Mono
