#!/usr/bin/env python3
"""
Extract seed data from epistemic DB and map to governance ingest format.
Writes data/seed.json with up to 200 events per agent.
"""

import json
import os
import random
import sqlite3
from datetime import datetime, timezone, timedelta

DB_PATH = "/home/node/.openclaw/epistemic/default.sqlite"
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed.json")

AGENTS = [
    "watson", "adlan", "argus", "dispatch",
    "content-scott", "content-apex", "felix-browser", "felix-build"
]

MODELS = [
    "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-3-5",
    "claude-sonnet-4-6", "claude-sonnet-4-6",  # weight toward sonnet-4-6
]

def agent_from_session(session_id: str) -> str:
    """Heuristic: map session prefixes to agent names."""
    sid = session_id.lower()
    if "compaction" in sid:
        return "watson"
    if "content" in sid:
        return random.choice(["content-scott", "content-apex"])
    if "felix" in sid:
        return random.choice(["felix-browser", "felix-build"])
    if "argus" in sid:
        return "argus"
    if "adlan" in sid:
        return "adlan"
    # Default: watson handles most sessions
    return "watson"


def estimate_tokens(text: str) -> tuple:
    """Rough token estimate: ~4 chars/token."""
    words = len(text.split())
    tokens_out = max(50, int(words * 1.3))
    tokens_in = max(100, tokens_out * random.randint(2, 8))
    return tokens_in, tokens_out


def main():
    events = []

    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Pull assistant outputs with decent length
        c.execute("""
            SELECT session_id, content, timestamp
            FROM decision_chunks
            WHERE role = 'assistant'
              AND length(content) > 80
            ORDER BY timestamp DESC
            LIMIT 2000
        """)
        rows = c.fetchall()
        conn.close()

        print(f"Found {len(rows)} usable assistant rows in epistemic DB")

        # Group by "agent" heuristic, cap at 200/agent
        per_agent: dict = {a: [] for a in AGENTS}

        for session_id, content, timestamp in rows:
            agent = agent_from_session(session_id)
            if len(per_agent[agent]) >= 200:
                continue

            tokens_in, tokens_out = estimate_tokens(content)
            latency_ms = random.randint(400, 6000)
            model = random.choice(MODELS)

            # Parse timestamp
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            except Exception:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            per_agent[agent].append({
                "agent": agent,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "output": content[:2000],  # cap at 2KB
                "timestamp": ts,
            })

        # Collect all
        for agent, agent_events in per_agent.items():
            events.extend(agent_events)
            print(f"  {agent}: {len(agent_events)} rows from DB")

    # Fill sparse agents with synthetic data
    now = datetime.now(timezone.utc)
    per_agent_count = {}
    for e in events:
        per_agent_count[e["agent"]] = per_agent_count.get(e["agent"], 0) + 1

    SYNTHETIC_OUTPUTS = {
        "argus": [
            "Security scan completed. No anomalies detected in the last 24 hours. All endpoints responding within SLA thresholds.",
            "Monitoring update: 3 new sessions observed, confidence 0.91. Alerting on token spike in session-4478.",
            "Audit log reviewed. No unauthorized access patterns. Rate limits stable.",
            "Drift check: argus fingerprint stable across 8 dimensions. Baseline confidence: 0.88.",
        ],
        "adlan": [
            "Repository scaffolded and pushed. Dockerfile configured for Cloud Run. GitHub Actions workflow active.",
            "Terraform apply completed successfully. Cloud Run service deployed at us-central1.",
            "FastAPI backend initialized with /api/ingest, /api/agents endpoints. Firestore connected.",
            "Build complete. Image pushed to GCR. Deployment pending GitHub Actions trigger.",
        ],
        "dispatch": [
            "Task routed to content-scott for LinkedIn post draft. ETA: 5 minutes.",
            "Handoff to adlan: project build request received, clarification phase started.",
            "Routing incoming Slack message to watson for triage. Priority: normal.",
            "Dispatch complete. Three agents notified. Await callbacks.",
        ],
        "content-scott": [
            "LinkedIn post draft ready. 280 words, first-person voice, no em dashes. Hook scored 8/10.",
            "Post revised per humanizer rules. Removed 'landscape', 'delve', em dash usage. Ready for review.",
            "Content brief processed. Five-post blitz scheduled across Mon–Fri slots.",
            "Image generated and uploaded to Postiz. Draft ID registered with reviewer.",
        ],
        "content-apex": [
            "Apex brand post drafted: 240 words, restaurant vertical focus. No hype language.",
            "Three LinkedIn posts completed for Apex parallel track. Hooks under 10 words.",
            "Campaign brief reviewed. Tone: confident, operator-friendly. Ready for scheduling.",
        ],
        "felix-browser": [
            "Navigated to target URL. Screenshot captured. Page title and content extracted.",
            "Browser automation complete. Form submitted, response 200. Extracted 12 data points.",
            "Web search executed. Top 5 results retrieved and summarized.",
        ],
        "felix-build": [
            "Build pipeline executed. Tests passed: 42/42. Docker image tagged and pushed.",
            "Lint and type checks complete. No critical issues found.",
            "CI run completed in 3m 42s. Deployment to staging triggered.",
        ],
        "watson": [
            "Task analyzed and routed. Summaries compiled. Memory coherence 0.87.",
            "Research brief complete. 5 high-signal market gaps identified with citations.",
            "Calendar reviewed. Motion tasks created for priority items. Reminders set.",
            "Webhook processed. n8n workflow triggered. Response dispatched.",
        ],
    }

    for agent in AGENTS:
        current = per_agent_count.get(agent, 0)
        needed = max(0, 50 - current)  # ensure at least 50 per agent
        if needed == 0:
            continue

        outputs = SYNTHETIC_OUTPUTS.get(agent, SYNTHETIC_OUTPUTS["watson"])
        print(f"  {agent}: adding {needed} synthetic rows (had {current})")

        for i in range(needed):
            output = outputs[i % len(outputs)]
            tokens_in, tokens_out = estimate_tokens(output)
            ts = (now - timedelta(minutes=random.randint(0, 10080))).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            events.append({
                "agent": agent,
                "model": random.choice(MODELS),
                "tokens_in": tokens_in + random.randint(-50, 200),
                "tokens_out": tokens_out + random.randint(-20, 100),
                "latency_ms": random.randint(400, 5000),
                "output": output,
                "timestamp": ts,
            })

    print(f"\nTotal seed events: {len(events)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Written to {OUT_PATH}")

    # Summary
    final_counts: dict = {}
    for e in events:
        final_counts[e["agent"]] = final_counts.get(e["agent"], 0) + 1
    print("\nPer-agent summary:")
    for a, cnt in sorted(final_counts.items()):
        print(f"  {a}: {cnt}")


if __name__ == "__main__":
    main()
