"""
Watson Governance Center - Backend API
Real-time LLM governance dashboard for OpenClaw agent ecosystem
"""

import asyncio
import hashlib
import json
import os
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
import math

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Try to import Firebase/Firestore
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/app/gcp-sa-key.json")
    if os.path.exists(sa_path) and not firebase_admin._apps:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIRESTORE_AVAILABLE = True
    print("✅ Firestore connected")
except Exception as e:
    FIRESTORE_AVAILABLE = False
    db = None
    print(f"⚠️  Firestore not available: {e}. Using in-memory store.")

app = FastAPI(title="Watson Governance Center", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store ──────────────────────────────────────────────────────────

# Rolling ring buffers
MAX_CALLS = 1000
recent_calls: deque = deque(maxlen=MAX_CALLS)

# Per-agent state
agent_state: Dict[str, Dict] = {}

# WebSocket connections
active_connections: Set[WebSocket] = set()

# Cost per 1K tokens (input, output) — approximate
MODEL_COSTS = {
    "claude-opus-4":        (0.015, 0.075),
    "claude-sonnet-4-5":    (0.003, 0.015),
    "claude-sonnet-4-6":    (0.003, 0.015),
    "claude-haiku-3-5":     (0.0008, 0.004),
    "gpt-4o":               (0.005, 0.015),
    "gpt-4o-mini":          (0.00015, 0.0006),
    "gemini-1.5-pro":       (0.00125, 0.005),
    "gemini-2.0-flash":     (0.0001, 0.0004),
    "default":              (0.003, 0.015),
}

KNOWN_AGENTS = [
    "watson", "adlan", "argus", "dispatch",
    "content-scott", "content-apex", "felix-browser", "felix-build"
]

HALLUCINATION_TYPES = ["factual", "temporal", "entity", "logical"]


# ── Evaluators ───────────────────────────────────────────────────────────────

def compute_confidence(text: str) -> float:
    """Simple heuristic confidence score based on hedging language."""
    if not text:
        return 0.5
    hedges = ["maybe", "perhaps", "might", "could be", "uncertain", "not sure",
              "i think", "probably", "possibly", "approximately", "around"]
    text_lower = text.lower()
    hedge_count = sum(1 for h in hedges if h in text_lower)
    word_count = max(len(text.split()), 1)
    hedge_ratio = hedge_count / (word_count / 10)
    return max(0.1, min(1.0, 1.0 - (hedge_ratio * 0.15)))


def detect_hallucination(text: str) -> Dict:
    """Classify hallucination risk and type."""
    if not text:
        return {"score": 0.0, "type": None, "risk": "low"}
    
    text_lower = text.lower()
    
    # Temporal hallucination signals
    temporal = ["in 2025", "recently", "last week", "yesterday", "just announced",
                "breaking:", "as of today"]
    # Entity hallucination signals  
    entity = ["ceo of", "founder of", "president of", "located at", "headquarters in"]
    # Factual
    factual = ["studies show", "research proves", "scientists found", "according to",
               "statistics show", "data shows"]
    # Logical
    logical = ["therefore", "thus", "consequently", "it follows", "necessarily",
               "always", "never", "all", "none"]
    
    scores = {
        "temporal": sum(1 for t in temporal if t in text_lower) * 0.15,
        "entity": sum(1 for e in entity if e in text_lower) * 0.12,
        "factual": sum(1 for f in factual if f in text_lower) * 0.08,
        "logical": sum(1 for l in logical if l in text_lower) * 0.05,
    }
    
    max_type = max(scores, key=scores.get)
    max_score = min(scores[max_type], 0.95)
    
    if max_score < 0.08:
        return {"score": max_score, "type": None, "risk": "low"}
    elif max_score < 0.25:
        return {"score": max_score, "type": max_type, "risk": "medium"}
    else:
        return {"score": max_score, "type": max_type, "risk": "high"}


def compute_fingerprint(text: str) -> List[float]:
    """
    8-dimensional semantic fingerprint for an agent's output.
    Dimensions: avg_sentence_len, vocab_diversity, hedge_ratio, 
                question_ratio, list_usage, code_usage, length_bucket, formality
    """
    if not text:
        return [0.0] * 8
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    words = text.split()
    
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1) / 50
    vocab_div = len(set(words)) / max(len(words), 1)
    hedge_count = sum(1 for w in ["maybe","perhaps","might","could","probably"] if w in text.lower())
    hedge_ratio = hedge_count / max(len(words) / 10, 1)
    question_ratio = text.count('?') / max(len(sentences), 1)
    list_usage = (text.count('\n-') + text.count('\n*') + text.count('\n1.')) / max(len(text) / 100, 1)
    code_usage = text.count('```') / max(len(text) / 500, 1)
    length_bucket = min(len(text) / 2000, 1.0)
    formality = 1 - (text.count("'") / max(len(words), 1) * 5)  # contractions → informal
    
    return [
        min(avg_sent_len, 1.0),
        vocab_div,
        min(hedge_ratio, 1.0),
        min(question_ratio, 1.0),
        min(list_usage, 1.0),
        min(code_usage, 1.0),
        length_bucket,
        max(0, min(formality, 1.0)),
    ]


def fingerprint_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two 8-dim fingerprints, normalized."""
    if not a or not b:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / math.sqrt(8)


# ── Bayesian Drift Detector ──────────────────────────────────────────────────

class BayesianDriftDetector:
    """
    Per-agent, per-dimension Bayesian Normal-Normal conjugate prior drift detection.
    Replaces Z-score approach with principled uncertainty-aware drift thresholds.
    """

    DEFAULT_MU = 0.5
    DEFAULT_VAR = 0.1

    def __init__(self):
        # {agent: {dim_name: {mu0, sigma0_sq, n, sum_x, sum_x_sq}}}
        self.priors: Dict[str, Dict[str, Dict]] = {}

    def _get_state(self, agent: str, dim: str) -> Dict:
        if agent not in self.priors:
            self.priors[agent] = {}
        if dim not in self.priors[agent]:
            self.priors[agent][dim] = {
                "mu0": self.DEFAULT_MU,
                "sigma0_sq": self.DEFAULT_VAR,
                "n": 0,
                "sum_x": 0.0,
                "sum_x_sq": 0.0,
            }
        return self.priors[agent][dim]

    def seed_from_history(self, agent: str, dim: str, values: List[float]):
        """Initialize prior from a list of historical observations."""
        if not values:
            return
        s = self._get_state(agent, dim)
        mu = sum(values) / len(values)
        var = (
            sum((x - mu) ** 2 for x in values) / len(values)
            if len(values) > 1
            else self.DEFAULT_VAR
        )
        s["mu0"] = mu
        s["sigma0_sq"] = max(var, 0.001)
        s["n"] = len(values)
        s["sum_x"] = sum(values)
        s["sum_x_sq"] = sum(x ** 2 for x in values)

    def update(self, agent: str, dim: str, value: float):
        """Incorporate a new observation into the running sufficient statistics."""
        s = self._get_state(agent, dim)
        s["n"] += 1
        s["sum_x"] += value
        s["sum_x_sq"] += value ** 2

    def get_posterior(self, agent: str, dim: str):
        """Return (posterior_mean, posterior_variance)."""
        s = self._get_state(agent, dim)
        n = s["n"]
        if n == 0:
            return s["mu0"], s["sigma0_sq"]
        obs_var = max(s["sum_x_sq"] / n - (s["sum_x"] / n) ** 2, 0.01)
        posterior_var = 1.0 / (1.0 / s["sigma0_sq"] + n / obs_var)
        posterior_mean = posterior_var * (
            s["mu0"] / s["sigma0_sq"] + s["sum_x"] / obs_var
        )
        return posterior_mean, posterior_var

    def is_drifting(self, agent: str, dim: str, value: float) -> bool:
        n = self._get_state(agent, dim)["n"]
        k = 3.0 if n < 10 else (2.5 if n < 50 else 2.0)
        mean, var = self.get_posterior(agent, dim)
        return abs(value - mean) > k * (var ** 0.5)

    def drift_score(self, agent: str, fingerprint: List[float]) -> float:
        """
        Compute an aggregate 0-1 drift score from an 8-dim fingerprint.
        Fraction of dimensions where is_drifting == True, weighted by deviation magnitude.
        """
        dims = [
            "avg_sent_len", "vocab_div", "hedge_ratio", "question_ratio",
            "list_usage", "code_usage", "length_bucket", "formality",
        ]
        if not fingerprint or len(fingerprint) != len(dims):
            return 0.0

        deviations = []
        for i, dim in enumerate(dims):
            mean, var = self.get_posterior(agent, dim)
            std = max(var ** 0.5, 1e-6)
            deviations.append(abs(fingerprint[i] - mean) / std)

        # Normalize: z-scores > 3 → full drift; mean of normalized scores → 0-1
        avg_z = sum(min(d / 3.0, 1.0) for d in deviations) / len(deviations)
        return min(avg_z, 1.0)

    def update_fingerprint(self, agent: str, fingerprint: List[float]):
        """Update all 8 dimensions from a fingerprint vector."""
        dims = [
            "avg_sent_len", "vocab_div", "hedge_ratio", "question_ratio",
            "list_usage", "code_usage", "length_bucket", "formality",
        ]
        for i, dim in enumerate(dims):
            if i < len(fingerprint):
                self.update(agent, dim, fingerprint[i])

    def calibration_confidence(self, agent: str) -> str:
        dims = self.priors.get(agent, {})
        if not dims:
            return "low"
        avg_n = sum(d["n"] for d in dims.values()) / len(dims)
        return "high" if avg_n >= 50 else ("medium" if avg_n >= 10 else "low")


# Global Bayesian detector instance
bayes_detector = BayesianDriftDetector()


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    in_rate, out_rate = MODEL_COSTS.get(model.lower(), MODEL_COSTS["default"])
    return (tokens_in / 1000 * in_rate) + (tokens_out / 1000 * out_rate)


def get_or_create_agent(agent_id: str) -> Dict:
    if agent_id not in agent_state:
        agent_state[agent_id] = {
            "id": agent_id,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "call_count": 0,
            "error_count": 0,
            "total_cost": 0.0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_latency_ms": 0,
            "fingerprint_baseline": None,
            "fingerprint_current": None,
            "fingerprint_drift": 0.0,
            "drift_status": "nominal",  # nominal / warning / critical
            "confidence_history": deque(maxlen=50),
            "hallucination_history": deque(maxlen=50),
            "models_used": defaultdict(int),
            "cost_history": deque(maxlen=100),  # (timestamp, cost)
        }
    return agent_state[agent_id]


# ── Pydantic Models ───────────────────────────────────────────────────────────

class IngestEvent(BaseModel):
    agent_id: str
    session_id: Optional[str] = None
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    output_text: Optional[str] = ""
    input_text: Optional[str] = ""
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    error: Optional[bool] = False


class IngestResponse(BaseModel):
    call_id: str
    confidence: float
    hallucination: Dict
    drift: float
    cost: float
    status: str


# ── WebSocket Manager ─────────────────────────────────────────────────────────

async def broadcast(event: Dict):
    """Broadcast event to all connected WebSocket clients."""
    if not active_connections:
        return
    msg = json.dumps(event)
    dead = set()
    for ws in active_connections:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    active_connections.difference_update(dead)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("/app/frontend/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "firestore": FIRESTORE_AVAILABLE,
        "agents": len(agent_state),
        "calls": len(recent_calls),
        "connections": len(active_connections),
    }


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(event: IngestEvent):
    """Main ingestion endpoint — receives agent call events, evaluates, stores, broadcasts."""
    ts = event.timestamp or datetime.now(timezone.utc).isoformat()
    call_id = hashlib.md5(f"{event.agent_id}{ts}{random.random()}".encode()).hexdigest()[:12]
    
    # Compute evaluations
    confidence = compute_confidence(event.output_text or "")
    hallucination = detect_hallucination(event.output_text or "")
    cost = compute_cost(event.model, event.tokens_in, event.tokens_out)
    
    # Fingerprint & Bayesian drift
    agent = get_or_create_agent(event.agent_id)
    fingerprint = compute_fingerprint(event.output_text or "")

    if agent["fingerprint_baseline"] is None and len(fingerprint) > 0:
        agent["fingerprint_baseline"] = fingerprint

    agent["fingerprint_current"] = fingerprint

    # Bayesian drift score (replaces raw fingerprint_distance threshold)
    drift = bayes_detector.drift_score(event.agent_id, fingerprint)
    # Update posterior with this new observation
    bayes_detector.update_fingerprint(event.agent_id, fingerprint)

    agent["fingerprint_drift"] = drift

    if drift < 0.15:
        drift_status = "nominal"
    elif drift < 0.30:
        drift_status = "warning"
    else:
        drift_status = "critical"
    agent["drift_status"] = drift_status
    
    # Update agent stats
    agent["last_seen"] = ts
    agent["call_count"] += 1
    if event.error:
        agent["error_count"] += 1
    agent["total_cost"] += cost
    agent["total_tokens_in"] += event.tokens_in
    agent["total_tokens_out"] += event.tokens_out
    agent["total_latency_ms"] += event.latency_ms
    agent["confidence_history"].append(confidence)
    agent["hallucination_history"].append(hallucination["score"])
    agent["models_used"][event.model] += 1
    agent["cost_history"].append((ts, cost))
    
    # Detect cost anomaly
    costs = [c for _, c in agent["cost_history"]]
    rolling_avg = sum(costs[:-1]) / max(len(costs) - 1, 1) if len(costs) > 1 else 0
    cost_anomaly = cost > (rolling_avg * 3) and cost > 0.01
    
    call_record = {
        "call_id": call_id,
        "agent_id": event.agent_id,
        "session_id": event.session_id,
        "model": event.model,
        "tokens_in": event.tokens_in,
        "tokens_out": event.tokens_out,
        "latency_ms": event.latency_ms,
        "output_text": (event.output_text or "")[:500],  # truncate for storage
        "input_text": (event.input_text or "")[:200],
        "timestamp": ts,
        "confidence": confidence,
        "hallucination_score": hallucination["score"],
        "hallucination_type": hallucination.get("type"),
        "hallucination_risk": hallucination.get("risk", "low"),
        "drift": drift,
        "drift_status": drift_status,
        "cost": cost,
        "cost_anomaly": cost_anomaly,
        "error": event.error,
        "fingerprint": fingerprint,
        "metadata": event.metadata or {},
    }
    
    recent_calls.appendleft(call_record)
    
    # Persist to Firestore
    if FIRESTORE_AVAILABLE and db:
        try:
            db.collection("agent_calls").document(call_id).set(call_record)
        except Exception as e:
            print(f"Firestore write error: {e}")
    
    # Broadcast via WebSocket
    await broadcast({
        "type": "call",
        "data": call_record,
    })
    
    # Also broadcast agent state update
    await broadcast({
        "type": "agent_update",
        "data": {
            "id": event.agent_id,
            "last_seen": ts,
            "call_count": agent["call_count"],
            "drift_status": drift_status,
            "fingerprint_drift": drift,
            "total_cost": agent["total_cost"],
            "avg_confidence": sum(agent["confidence_history"]) / len(agent["confidence_history"]) if agent["confidence_history"] else 0,
        }
    })
    
    return IngestResponse(
        call_id=call_id,
        confidence=confidence,
        hallucination=hallucination,
        drift=drift,
        cost=cost,
        status="ok",
    )


@app.get("/api/stats")
async def get_stats():
    """Global stats summary."""
    total_calls = sum(a["call_count"] for a in agent_state.values())
    total_cost = sum(a["total_cost"] for a in agent_state.values())
    
    # Hallucination type breakdown from recent calls
    type_counts = defaultdict(int)
    for call in list(recent_calls)[:200]:
        ht = call.get("hallucination_type")
        if ht:
            type_counts[ht] += 1
    
    # Rolling 24hr heatmap data (agent → hour → avg hallucination)
    now = datetime.now(timezone.utc)
    heatmap = defaultdict(lambda: defaultdict(list))
    for call in list(recent_calls):
        ts_str = call.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if (now - ts).total_seconds() < 86400:
                hour = ts.hour
                heatmap[call["agent_id"]][hour].append(call.get("hallucination_score", 0))
        except Exception:
            pass
    
    heatmap_out = {}
    for agent, hours in heatmap.items():
        heatmap_out[agent] = {
            str(h): sum(v) / len(v) for h, v in hours.items()
        }
    
    return {
        "total_calls": total_calls,
        "total_cost": total_cost,
        "total_agents": len(agent_state),
        "active_connections": len(active_connections),
        "hallucination_types": dict(type_counts),
        "heatmap": heatmap_out,
        "calls_last_hour": sum(
            1 for c in recent_calls
            if (datetime.now(timezone.utc) - 
                datetime.fromisoformat(c.get("timestamp", "2000-01-01").replace("Z", "+00:00"))
            ).total_seconds() < 3600
        ),
    }


@app.get("/api/agents")
async def get_agents():
    """Per-agent health and state."""
    out = []
    for agent_id, a in agent_state.items():
        conf_history = list(a["confidence_history"])
        hall_history = list(a["hallucination_history"])
        cost_history = [(ts, c) for ts, c in list(a["cost_history"])[-20:]]
        
        out.append({
            "id": agent_id,
            "name": agent_id,
            "first_seen": a["first_seen"],
            "last_seen": a["last_seen"],
            "call_count": a["call_count"],
            "error_count": a["error_count"],
            "error_rate": a["error_count"] / max(a["call_count"], 1),
            "total_cost": a["total_cost"],
            "avg_latency_ms": a["total_latency_ms"] / max(a["call_count"], 1),
            "avg_confidence": sum(conf_history) / max(len(conf_history), 1),
            "avg_hallucination": sum(hall_history) / max(len(hall_history), 1),
            "drift_status": a["drift_status"],
            "fingerprint_drift": a["fingerprint_drift"],
            "fingerprint_current": a["fingerprint_current"],
            "fingerprint_baseline": a["fingerprint_baseline"],
            "confidence_sparkline": conf_history[-10:],
            "models_used": dict(a["models_used"]),
            "cost_history": cost_history[-10:],
            "calibration_confidence": bayes_detector.calibration_confidence(agent_id),
        })
    
    # Include known agents that haven't been seen yet
    for agent_id in KNOWN_AGENTS:
        if agent_id not in agent_state:
            out.append({
                "id": agent_id,
                "name": agent_id,
                "call_count": 0,
                "error_count": 0,
                "error_rate": 0,
                "total_cost": 0,
                "avg_latency_ms": 0,
                "avg_confidence": 0,
                "avg_hallucination": 0,
                "drift_status": "nominal",
                "fingerprint_drift": 0,
                "status": "never_seen",
                "calibration_confidence": bayes_detector.calibration_confidence(agent_id),
            })
    
    return out


@app.get("/api/calls")
async def get_calls(limit: int = Query(50, le=200), agent_id: Optional[str] = None):
    """Recent call records."""
    calls = list(recent_calls)
    if agent_id:
        calls = [c for c in calls if c.get("agent_id") == agent_id]
    return calls[:limit]


@app.get("/api/drift")
async def get_drift():
    """Drift fingerprints for all agents."""
    return [
        {
            "agent_id": agent_id,
            "drift_status": a["drift_status"],
            "fingerprint_drift": a["fingerprint_drift"],
            "fingerprint_current": a["fingerprint_current"],
            "fingerprint_baseline": a["fingerprint_baseline"],
            "call_count": a["call_count"],
        }
        for agent_id, a in agent_state.items()
    ]


@app.get("/api/costs")
async def get_costs():
    """Cost breakdown per agent and model."""
    by_agent = {}
    by_model = defaultdict(float)
    
    for agent_id, a in agent_state.items():
        by_agent[agent_id] = {
            "total": a["total_cost"],
            "by_model": dict(a["models_used"]),
        }
        for model, count in a["models_used"].items():
            # Estimate cost per model usage
            in_rate, out_rate = MODEL_COSTS.get(model, MODEL_COSTS["default"])
            estimated = count * 500 * (in_rate + out_rate) / 1000
            by_model[model] += estimated
    
    total = sum(a["total_cost"] for a in agent_state.values())
    
    # Burn rate (cost per hour based on last hour)
    hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    hour_cost = sum(
        c.get("cost", 0) for c in recent_calls
        if datetime.fromisoformat(c.get("timestamp", "2000-01-01").replace("Z", "+00:00")) > hour_ago
    )
    
    return {
        "total": total,
        "by_agent": by_agent,
        "by_model": dict(by_model),
        "burn_rate_per_hour": hour_cost,
        "burn_rate_per_day_estimate": hour_cost * 24,
    }


@app.websocket("/api/live")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time event stream via WebSocket."""
    await websocket.accept()
    active_connections.add(websocket)
    
    # Send initial state
    try:
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "agents": list(agent_state.keys()),
                "recent_calls": list(recent_calls)[:20],
                "known_agents": KNOWN_AGENTS,
            }
        }))
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Handle ping
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat", "ts": datetime.now(timezone.utc).isoformat()}))
            except WebSocketDisconnect:
                break
    except Exception:
        pass
    finally:
        active_connections.discard(websocket)


# ── Seed from epistemic DB data ───────────────────────────────────────────────

SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed.json")

FINGERPRINT_DIMS = [
    "avg_sent_len", "vocab_div", "hedge_ratio", "question_ratio",
    "list_usage", "code_usage", "length_bucket", "formality",
]


async def load_seed_data():
    """
    Load data/seed.json and:
    1. Initialize Bayesian priors from historical fingerprints.
    2. Ingest events so the dashboard shows real call history.
    """
    await asyncio.sleep(1)

    if not os.path.exists(SEED_PATH):
        print(f"⚠️  Seed file not found at {SEED_PATH}, skipping seed load")
        return

    try:
        with open(SEED_PATH) as f:
            seed_events = json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load seed.json: {e}")
        return

    print(f"📂 Loading {len(seed_events)} seed events from epistemic DB...")

    # Phase 1: build Bayesian priors from all fingerprints (no ingestion yet)
    per_agent_fingerprints: Dict[str, List[List[float]]] = {}
    for ev in seed_events:
        agent = ev.get("agent", "watson")
        fp = compute_fingerprint(ev.get("output", ""))
        if agent not in per_agent_fingerprints:
            per_agent_fingerprints[agent] = []
        per_agent_fingerprints[agent].append(fp)

    for agent, fingerprints in per_agent_fingerprints.items():
        for dim_idx, dim_name in enumerate(FINGERPRINT_DIMS):
            values = [fp[dim_idx] for fp in fingerprints if len(fp) > dim_idx]
            bayes_detector.seed_from_history(agent, dim_name, values)

    print(f"✅ Bayesian priors seeded for {len(per_agent_fingerprints)} agents")

    # Phase 2: ingest a subset (up to 40 most recent per agent) to populate dashboard
    by_agent: Dict[str, list] = {}
    for ev in seed_events:
        a = ev.get("agent", "watson")
        by_agent.setdefault(a, []).append(ev)

    ingested = 0
    for agent, evs in by_agent.items():
        # Most recent first (seed.json is newest-first from DB query)
        for ev in evs[:40]:
            event = IngestEvent(
                agent_id=ev.get("agent", "watson"),
                session_id=f"seed-{agent}",
                model=ev.get("model", "claude-sonnet-4-6"),
                tokens_in=ev.get("tokens_in", 500),
                tokens_out=ev.get("tokens_out", 150),
                latency_ms=ev.get("latency_ms", 1500),
                output_text=ev.get("output", "")[:500],
                timestamp=ev.get("timestamp"),
                error=False,
            )
            await ingest(event)
            ingested += 1

    print(f"✅ Ingested {ingested} seed events. Dashboard ready.")

    # Log calibration confidence per agent
    for agent in per_agent_fingerprints:
        conf = bayes_detector.calibration_confidence(agent)
        print(f"   {agent}: calibration_confidence={conf}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_seed_data())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
