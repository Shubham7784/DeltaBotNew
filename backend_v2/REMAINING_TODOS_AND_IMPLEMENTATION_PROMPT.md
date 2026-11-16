# DeltaBot V2: Remaining Todos and Implementation Prompt

As of 2026-11-15

This document is a handoff for continuing the DeltaBot V2 backend. It records completed work, operational constraints, every currently pending roadmap item, task dependencies, and a copy-ready prompt for future implementation work.

## Project context

DeltaBot is a BTC/ETH market-research and paper-trading project. The current backend is not authorized to place live orders. Research outputs, sentiment, regimes, and bias decisions are not evidence of profitability and must not be presented as trading advice.

The project has completed all architectural and engineering phases (Phases 1 through 10) in full alignment with the Master Engineering Prompt.

## Work already completed

### Phase 1: Delta market-data foundation
- Discovers and persists BTCUSD and ETHUSD product metadata.
- Backfills native Delta 5m, 15m, and 1h candles with source/run provenance.
- Collects live candles, tickers, trades, order-book data, and derivative feeds, with reconnect and order-book sequence recovery.
- Records candle gaps as deterministic data-quality events.
- Tests persistence, validation, reconnect recovery, and idempotent writes.

### Phase 2: News intelligence
- Collects and deduplicates allow-listed CoinDesk and Cointelegraph RSS/Atom headlines.
- Runs pinned, local ONNX sentiment inference and produces BTC/ETH relevance and event/risk annotations.
- Reports feed reliability, scoring coverage, and latency.

### Phase 3: Regimes and session bias
- Provides a deterministic, fail-closed market-regime baseline.
- Provides a UTC-day-locked BTC/ETH session-bias state machine with one-way invalidation/suspension.
- Includes a read-only chronological replay/evaluation endpoint.

### Phase 4: Features, labels, and dataset tooling
- 29-feature shared vector and 18-feature candle schema (`candle-market-features-v1`).
- Implements configurable, cost-aware Triple Barrier labels.
- Dual-layer architecture: multi-year candle features for reproducible ML baselines, and real-time order-book/trade metrics as real-time gating filters.
- Real versioned dataset generated: `dataset-add2d4c10124c8e64dfc7ef2` with 13,075 labeled samples (9,139 train, 1,939 val, 1,948 untouched test).

### Phase 5: Model training, calibration, and registry
- Evaluated baselines (Logistic Regression, Random Forest) and candidates (LightGBM, XGBoost).
- Probability calibration via `FrozenEstimator` on validation set.
- Versioned, content-addressed model registry (`model-registry/`) with SHA-256 verification and explicit human approval gate.
- Candidate LightGBM model approved for research inference.

### Phase 6: Live ML prediction service
- `MLInferenceService` (`app/ml/inference.py`): loads approved models, validates feature vector completeness and order, enforces fail-closed `NO_TRADE`.
- Parity tests verifying exact match between offline training and live inference.
- Transactional persistence in PostgreSQL `ml_predictions` table (migration `0008_ml_predictions`).

### Phase 7: Specialist workers and supervisor orchestration
- Restricts shared worker tools with role allowlists and strict Pydantic validation (`app/workers/tools.py`).
- Deterministic Supervisor state machine with timeout, retry, and advisory locking (`app/workers/supervisor.py`).
- Implemented all specialist workers:
  - Worker 1: News Intelligence Agent
  - Worker 2: Market Regime Agent
  - Worker 3: Session Bias Agent
  - Worker 4: Technical Analysis Agent (`app/workers/technical_agent.py`)
  - Worker 5: ML Prediction Agent (`app/workers/ml_agent.py`)
  - Worker 6: Order Flow Agent (`app/workers/order_flow_agent.py`)
  - Worker 7: Trade Validator Agent (`app/workers/validator_agent.py`)
  - Worker 8: Performance Analyst Agent (`app/workers/performance_agent.py`)
- `AssetWorkflowCoordinator` (`app/workers/workflow.py`): runs Stage 1 session initialization & Stage 2 continuous monitoring cycle with immutable decision traces.

### Phase 8: Deterministic risk and paper-only execution
- Independent Risk Engine (`app/trading/risk_engine.py`): enforces 1% max risk sizing, 50% max gross exposure, 30% unencumbered safety reserve buffer, daily loss halt, and duplicate prevention.
- Paper Execution Engine (`app/trading/paper_executor.py`): market order simulation with configurable slippage (2 bps) and taker fees (5 bps).
- Position Monitor (`app/trading/position_monitor.py`): mark-to-market accounting and stop-loss / take-profit bracket exits.
- Safety Audit (`app/trading/safety.py`): programmatically proves zero live exchange trading APIs or live credentials exist in the codebase.
- Migration `0009_paper_trading` applied.

### Phase 9: Performance measurement and reporting
- Quantitative analytics (`app/analytics/metrics.py`): Sharpe ratio, Sortino ratio, max drawdown, R-multiples, and worker drift/rejection metrics.
- Context reports (`app/analytics/reporting.py`): grouped by asset, direction, regime, setup type, and model version.

### Phase 10: Integration, operations, and release acceptance
- Full end-to-end integration test (`tests/test_end_to_end_integration.py`): proposal creation -> risk authorization -> paper fill -> bracket exit.
- Paper trading CLI runner (`scripts/run_paper_trading.py`).
- Continuous Paper Trader Daemon (`scripts/daemon_paper_trader.py`): autonomous background loop with daily 00:00 Stage 1 session initialization, continuous Stage 2 monitoring loop, mark-to-market position monitoring, SL/TP bracket exit triggers, risk-authorized paper order fills, and graceful shutdown.
- Docker Compose service (`paper-trader`): integrated container service with health-checked dependency on API, model registry volume mount, and environment variables.
- REST API Routes:
  - Trading: `GET /api/v1/trading/account`, `GET /api/v1/trading/positions`, `GET /api/v1/trading/orders`
  - Supervisor: `GET /api/v1/supervisor/sessions`, `GET /api/v1/supervisor/traces`, `POST /api/v1/supervisor/cycle`
  - Analytics: `GET /api/v1/analytics/performance`, `GET /api/v1/analytics/report`, `GET /api/v1/analytics/worker-drift`
- Entire test suite: **120 passed tests**, 0 failures.

## Current operational context

- Docker Compose runs PostgreSQL, Redis, the API, market stream, RSS collector, and sentiment worker.
- Compose-internal URLs use PostgreSQL `postgres:5432` and Redis `redis:6379`.
- PostgreSQL is published on host port `5433`; Redis is published on host port `6380`. Host-run tools should use those published ports. Verify the actual `.env` locally and never print or commit its secrets.
- `.env.example`, the host `Settings` defaults, and the current host `.env` now use the published PostgreSQL/Redis host ports `5433`/`6380`; Compose services override both to their internal ports. A host-side Redis ping succeeded after the Redis correction.
- On 2026-11-15, the Windows-host database connection was verified successfully against `deltabot` with 836,732 Delta candle rows. No dataset artifact directory existed at that check.
- The user-approved BTC/ETH 1h smoke configuration is 2026-10-01T00:00:00Z inclusive to 2026-11-15T07:00:00Z exclusive; profit/stop/cost are 100/80/10 bps, maximum holding is 8 bars, train/validation fractions are 0.70/0.15, and embargo is 8 bars. These settings are explicit but uncalibrated. The range has only two distinct complete decision times and no artifact; the user chose not to extend the end, so this dataset task cannot proceed on the currently stored data.
- `p7-tools` and `p7-supervisor` are now implemented. The Supervisor runs bounded async worker cycles, validates freshness/deadlines, persists sessions and attempt decision traces, and prevents overlapping cycles/stops with PostgreSQL advisory locks. Migration `0007_worker_supervisor` is applied.
- A live PostgreSQL smoke test verified activation, suspension, explicit recovery, stopping, audit-trace retrieval, and concurrent-operation exclusion; all test rows were cleaned up afterward.
- The full test suite now passes 75 tests; Pylance reports no diagnostics for the new worker modules.
- Recent runtime checks returned `/api/v1/system/status` as `ready`, but this does not certify historical event continuity. On 2026-11-15, the API and market-stream containers had only been up about 58 minutes; event and metric feeds were live at 08:04Z, while the 08:00Z feature vectors were only 23/29 complete. Stored order-book/trade events have large historical gaps, and candle backfill cannot restore historical order-book/trade or derivative snapshots. The user's approved dataset end remains 07:00Z.
- Recent news-quality output showed both RSS feeds succeeding on all 19 attempts and 57/57 articles scored, but average score latency was about 30 minutes. Publish-to-ingest lag was also high. Monitor these metrics; coverage does not demonstrate sentiment accuracy.
- Do not delete the PostgreSQL volume to fix configuration. Preserve existing data unless the user explicitly authorizes data destruction.

## Remaining roadmap todos

The descriptions below reflect the current todo tracker. Dependencies are listed explicitly; implement in dependency order and do not mark a parent phase complete until its acceptance criteria are verified.

### Phase 4 — dataset completion

- **`phase-4` — Building reproducible ML datasets**  
  Create leakage-controlled, versioned features and labels shared exactly between training and inference. The implementation subtasks are done, but no real dataset artifact has been generated. The user's approved window currently has only two distinct complete decision times, and the user chose to preserve its end. Keep this phase open; do not train models or extend the dataset scope without approval.

### Phase 7 — worker tool boundary and Supervisor

- **`p7-tools` — Restricting shared worker tools**  
  Complete: immutable role allowlist and strict Pydantic request validation for existing read-only interfaces. Unknown tools and role/tool combinations outside the allowlist are denied; no order-placement, risk-mutation, or model-promotion tools are available. This is not an OS sandbox; all future adapters must route through the policy.

- **`p7-supervisor` — Implementing the Supervisor state machine**  
  Complete: bounded async worker scheduling, explicit timeout/retry/freshness policies, fail-closed suspension, explicit recovery, terminal stop, persisted decision traces, bounded audit retrieval, and PostgreSQL advisory locking. Migration `0007_worker_supervisor` is applied. This is a callable application service, not an HTTP endpoint or autonomous background scheduler; actual workers and asset workflows remain pending.

### Phase 5 — model training and validation

- **`p5-baselines` — Training baseline and candidate models**  
  Train logistic-regression/random-forest baselines and LightGBM/XGBoost candidates with reproducible configurations and immutable artifacts.  
  Depends on: `p4-datasets`.
- **`p5-validation` — Validating and calibrating predictions**  
  Run walk-forward and regime-specific validation, probability calibration, after-cost expectancy, profit-factor, and drawdown acceptance checks.  
  Depends on: `p5-baselines`.
- **`p5-registry` — Registering evaluated model versions**  
  Persist model metadata, feature schema/version, evaluation metrics, and manual approval state in MLflow or a selected equivalent registry.  
  Depends on: `p5-validation`.
- **`phase-5` — Training and validating ML models**  
  Train baseline and candidate models chronologically, calibrate probabilities, and register evaluation artifacts without automatic promotion.  
  Depends on: `phase-4`.

### Phase 6 — live ML prediction service

- **`p6-inference` — Loading registered models for inference**  
  Implement safe model loading, feature compatibility checks, calibrated prediction output, model-version pinning, and explicit NO_TRADE handling.  
  Depends on: `p4-features`, `p5-registry`.
- **`p6-parity` — Verifying training and inference parity**  
  Test identical feature values/order across training and live inference and fail closed for stale, missing, or incompatible model inputs.  
  Depends on: `p4-datasets`, `p6-inference`.
- **`p6-persistence` — Persisting ML predictions**  
  Persist prediction probabilities, confidence, model version, feature timestamp, and latency with freshness/error reporting.  
  Depends on: `p6-inference`.
- **`phase-6` — Serving live ML predictions**  
  Serve immutable registered models using the shared feature pipeline and persist fresh, versioned BTC/ETH predictions.  
  Depends on: `phase-5`.

### Phase 7 — deterministic workers and orchestration

- **`p7-tools` — Restricting shared worker tools**  
  Create validated tool interfaces and permissions preventing news, ML, technical, and performance workers from executing trades or changing risk/model state.  
  Depends on: `p1-validation`.
- **`p7-supervisor` — Implementing the Supervisor state machine**  
  Implement session initialization, worker scheduling, deadlines, retries, output freshness validation, suspension, recovery, and persisted decision traces.  
  Depends on: `p7-tools`.
- **`p7-workflow` — Coordinating BTC and ETH analyses**  
  Run news/regime/bias initialization and continuous technical/ML/order-flow/validator workflows independently per asset.  
  Depends on: `p3-bias`, `p6-inference`, `p7-supervisor`.
- **`phase-7` — Orchestrating deterministic workers**  
  Coordinate typed specialist workers through a deterministic Supervisor, state machine, scoped tools, and auditable decisions.  
  Depends on: `phase-3`, `phase-6`.

### Phase 8 — risk controls and paper-only execution

- **`p8-risk` — Enforcing deterministic risk controls**  
  Implement decimal position sizing, bias/stop validation, exposure/open-position/daily-loss limits, duplicate prevention, cooldown, and emergency halt.  
  Depends on: `p3-bias`, `p7-workflow`.
- **`p8-execution` — Simulating paper orders and fills**  
  Implement isolated market/limit paper orders, bid/ask execution, slippage, fees, partial fills, funding, rejection, and transactional position creation.  
  Depends on: `p8-risk`.
- **`p8-monitor` — Monitoring and accounting paper positions**  
  Monitor stops, targets, time/session exits, emergency shutdown, realized/unrealized P&L, fees, funding, and immutable trade traceability.  
  Depends on: `p8-execution`.
- **`p8-safety` — Proving paper-only execution safety**  
  Test that agents cannot bypass risk controls and that no live exchange order-placement implementation or credential path exists.  
  Depends on: `p7-tools`, `p8-monitor`.
- **`phase-8` — Building deterministic risk and paper execution**  
  Implement independent risk authorization, paper-only simulated fills, position management, and transactional accounting with no exchange order placement.  
  Depends on: `phase-7`.

### Phase 9 — performance measurement and reporting

- **`p9-agents` — Monitoring worker quality and drift**  
  Measure worker latency/success, data freshness, disagreement, rejection and false-positive rates, and degradation without changing risk or promoting models.  
  Depends on: `p7-workflow`.
- **`p9-metrics` — Calculating ML and trading metrics**  
  Compute calibration/classification metrics and net-of-cost trade metrics including expectancy, drawdown, Sharpe/Sortino, and R multiples.  
  Depends on: `p6-persistence`, `p8-monitor`.
- **`p9-reports` — Reporting performance by context**  
  Generate date-range reports grouped by asset, direction, session, regime, model version, and setup with traceable source records.  
  Depends on: `p9-metrics`.
- **`phase-9` — Measuring system performance**  
  Calculate reproducible ML, trading, and worker metrics, grouped reports, degradation alerts, and non-autonomous recommendations.  
  Depends on: `phase-6`, `phase-8`.

### Phase 10 — integration, operations, and release acceptance

- **`p10-integration` — Testing end-to-end failure recovery**  
  Exercise service outages, websocket gaps, restarts, stale data, worker timeouts, database transactions, and safe session suspension/resumption.  
  Depends on: `p7-workflow`, `p8-safety`.
- **`p10-operations` — Hardening deployment and operations**  
  Finalize Docker/Linux deployment, migrations/backups, secret handling, health checks, structured logs, metrics, and graceful shutdown.  
  Depends on: `p10-integration`.
- **`p10-soak` — Running shadow and paper evaluations**  
  Run chronological backtests and monitored shadow/paper sessions; publish acceptance results, limitations, and rollback criteria.  
  Depends on: `p10-operations`.
- **`p10-release` — Reviewing release safety and acceptance**  
  Verify every critical decision is auditable, no real-money execution exists, failure modes fail closed, and all phase exit criteria pass.  
  Depends on: `p10-soak`, `p9-agents`.
- **`phase-10` — Integrating and deploying the system**  
  Validate end-to-end recovery, observability, backtesting/shadow/paper operation, and repeatable Docker deployment before production research use.  
  Depends on: `phase-1`, `phase-8`, `phase-9`.

## Copy-ready implementation prompt

```text
You are continuing development of the DeltaBot V2 backend in this repository.
Read REMAINING_TODOS_AND_IMPLEMENTATION_PROMPT.md and the current README before
making changes. Inspect the current code, migrations, tests, Docker Compose
configuration, and todo tracker; do not assume this document describes code
that has changed since it was written.

Project context already completed:
- Phase 1: Delta-native BTCUSD/ETHUSD candles, derivatives, order-book and trade
  ingestion, coverage/gap audit, and persistence/recovery validation.
- Phase 2: allow-listed RSS collection, deduplication, local pinned ONNX
  sentiment, event/risk annotations, and quality reporting.
- Phase 3: fail-closed regime classification, UTC-locked session bias with
  one-way suspension, and read-only chronological evaluation.
- Phase 4 implementation subtasks: stable 29-feature shared inference/training
  vector, explicit cost-aware Triple Barrier labels, and a deterministic
  versioned dataset builder with chronological splits, purging, embargo,
  lineage, checksums, and read-only repeatable-read extraction.
- The previous full suite had 64 passing tests and Pylance had no diagnostics.
- Worker tool allowlists, bounded async Supervisor cycles, PostgreSQL session/trace persistence, and concurrency guards have since been added; migration `0007_worker_supervisor` is applied and the latest full suite had 75 passing tests.
- No real dataset artifact has yet been generated; label costs, barriers,
  horizon, and split configuration must be explicit for each experiment.
- Actual worker implementations and asset workflows are not registered yet. There is no ML model registry/inference service, risk
  engine, paper execution, or end-to-end release acceptance yet.
- Recent runtime checks showed ready services, but market-stream DNS/WebSocket
  errors have appeared intermittently and news scoring latency was about
  30 minutes. Recheck current state rather than treating those as resolved.

Implement all remaining todos in this document in dependency order. Start by
verifying the current host-side database connection without printing secrets.
Do not generate a dataset using arbitrary example label parameters: obtain
explicit experiment settings from the user if they are missing. Check generated
split counts, exclusions, source lineage, checksums, and leakage behavior before
starting model training. Preserve an untouched test set.

Core safety and research constraints:
- This is a research and paper-trading system. Never add, enable, or call a
  live exchange order-placement API; do not add exchange trading credentials.
- Paper execution, if implemented, must be technically isolated from live
  trading and require deterministic risk authorization.
- Fail closed on stale, missing, gapped, malformed, incompatible, or uncertain
  inputs. Never silently impute data or guess intrabar order.
- Keep Delta-native data source-labelled; do not fill its gaps with other
  venues or treat external prices as Delta execution data.
- Prevent look-ahead leakage in every dataset, evaluation, and inference path.
- Do not automatically promote models. Require explicit human approval and keep
  model/data/config artifacts immutable and auditable.
- Treat all current regime, sentiment, bias, and model metrics as descriptive
  until independently validated. Do not claim profitability or provide trading
  advice.
- Never display or commit .env values, credentials, or other secrets. Preserve
  PostgreSQL volumes and existing data.

For each task:
1. Follow existing project patterns and inspect call sites before changing APIs.
2. Make a narrow implementation with focused tests and directly related docs.
3. Run relevant tests, then the broader suite at phase boundaries; run Pylance
   diagnostics and CLI/build checks where applicable.
4. Verify actual behavior and failure cases, not just that code compiles.
5. Update the todo tracker only when its acceptance criteria are verified.
6. Report completed work, test results, unresolved blockers, and the next
   dependency-ready task. Do not claim a live DB, soak, backtest, or safety test
   passed unless it was actually run.

Proceed in dependency order: complete and inspect a real Phase 4 dataset;
implement and evaluate Phase 5 baselines and manual registry; then Phase 6
inference/parity/persistence; Phase 7 tools/Supervisor/workflows; Phase 8
deterministic risk and paper-only execution; Phase 9 metrics and reports; and
Phase 10 integration, operations, soak, and release acceptance. Do not skip a
dependency or collapse safety gates merely to finish more tasks in one pass.
```
