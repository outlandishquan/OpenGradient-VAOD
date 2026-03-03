# Verifiable AI Output Demo

A minimal, production-quality FastAPI application demonstrating **verifiable AI inference** using the [OpenGradient Python SDK](https://docs.opengradient.ai/developers/sdk/llm.html) with TEE (Trusted Execution Environment) verification and x402 on-chain settlement.

---

## Project Structure

```
Opengradient VAOD/
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
├── README.md             # ← You are here
└── app/
    ├── __init__.py
    ├── config.py          # Settings & env loading
    ├── og_client.py       # OpenGradient client wrapper
    ├── inference.py       # Inference service layer
    ├── routes.py          # FastAPI route handlers
    ├── main.py            # Application entrypoint
    └── templates/
        └── index.html     # Minimal UI
```

---

## Prerequisites

- **Python 3.10+**
- A **Base Sepolia wallet** funded with `$OPG` testnet tokens
  - Get tokens: <https://faucet.opengradient.ai>

---

## Setup & Run

### 1. Install dependencies

```bash
cd "Opengradient VAOD"
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your private key:

```env
OG_PRIVATE_KEY=0xYOUR_ACTUAL_PRIVATE_KEY
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the UI

Navigate to **http://localhost:8000** in your browser.

---

## How It Works — Verifiable Inference

```
┌─────────────┐     POST /infer      ┌──────────────┐
│   Browser    │ ──────────────────▶  │  FastAPI App  │
│   (prompt)   │                      │               │
└─────────────┘                      │  inference.py │
                                      │       │       │
                                      │       ▼       │
                                      │  og_client.py │
                                      └───────┬───────┘
                                              │
                              client.llm.chat() via x402
                                              │
                                              ▼
                                ┌─────────────────────────┐
                                │   OpenGradient Network   │
                                │  ┌───────────────────┐  │
                                │  │   TEE Node         │  │
                                │  │  (Intel TDX)       │  │
                                │  │   ┌─────────────┐  │  │
                                │  │   │ LLM Provider │  │  │
                                │  │   │ (OpenAI, etc)│  │  │
                                │  │   └─────────────┘  │  │
                                │  └───────────────────┘  │
                                │                         │
                                │  x402 settlement        │
                                │  on Base Sepolia        │
                                └─────────────────────────┘
```

1. **User submits a prompt** through the web UI.
2. **FastAPI backend** calls `client.llm.chat()` from the OpenGradient SDK.
3. The SDK **signs an x402 payment** with your wallet's private key.
4. The request is routed to a **TEE node** on the OpenGradient network, which runs inside an Intel TDX enclave providing hardware-level attestation.
5. The TEE node forwards the prompt to the upstream **LLM provider** (OpenAI, Anthropic, Google, or xAI).
6. The response is returned with:
   - `payment_hash` — the x402 payment transaction hash on Base Sepolia
   - `transaction_hash` — the on-chain settlement transaction
   - `finish_reason` — why the model stopped generating
7. The **UI displays a Verification badge** based on the presence of `payment_hash`.

### What makes this "verifiable"?

- **TEE execution**: Every inference runs inside a Trusted Execution Environment, giving cryptographic proof that the routing and verification code executed correctly.
- **x402 on-chain settlement**: Each call produces an on-chain receipt (`payment_hash`) that anyone can verify on the [OpenGradient Block Explorer](https://explorer.opengradient.ai).
- **No invented claims**: The demo only surfaces fields actually returned by the SDK.

---

## Settlement Modes

| Mode              | Description                                        |
|-------------------|----------------------------------------------------|
| `SETTLE`          | Most privacy-preserving — no data hashes on-chain  |
| `SETTLE_METADATA` | Full transparency — all input/output data on-chain |
| `SETTLE_BATCH`    | Cost-efficient — batches multiple calls together   |

---

## API Endpoints

| Method | Path      | Description                    |
|--------|-----------|--------------------------------|
| GET    | `/`       | Web UI                         |
| POST   | `/infer`  | Run inference (JSON body)      |
| GET    | `/health` | Health check                   |

### POST `/infer` — Request Body

```json
{
  "prompt": "What is 2+2?",
  "model": "GPT_4O",
  "settlement_mode": "SETTLE_METADATA"
}
```

---

## Links

- [OpenGradient Docs](https://docs.opengradient.ai)
- [Python SDK Reference](https://docs.opengradient.ai/api_reference/python_sdk/)
- [Testnet Faucet](https://faucet.opengradient.ai)
- [Block Explorer](https://explorer.opengradient.ai)
