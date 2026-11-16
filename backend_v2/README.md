# DeltaBot — Market Data and News Ingestion

The repository currently implements the Phase 1 market-data foundation, Phase 2 RSS news collection and sentiment classification, a deterministic Phase 3 market-regime baseline, and Phase 4 shared market features and research dataset tooling. It does not contain live trading, paper execution, worker orchestration, or model-promotion code.

## Run locally

1. Copy `.env.example` to `.env`; keep the default India endpoints or configure the appropriate regional endpoints:
   - REST: `https://api.india.delta.exchange`
   - Public WebSocket: `wss://public-socket.india.delta.exchange`
2. Start dependencies, API, and the public market stream with `docker compose up --build`.
3. Check `GET /health/live`, `GET /health/ready`, or `GET /api/v1/system/status`.
4. Open `/docs` for the FastAPI OpenAPI page.
5. Inspect the RSS collector with `docker compose logs -f news-collector`.

The Redis container listens on port `6380` on the host by default, while the app containers connect to Redis and PostgreSQL over the Compose network. For running the backend directly on the host, `REDIS_URL` must use the published host port and stay aligned with `REDIS_HOST_PORT`; the Compose services use the internal Redis hostname and port regardless.

Initialize and inspect data using the scripts from the repository root:

```powershell
alembic upgrade head
python -m scripts.refresh_products
python -m scripts.backfill_candles --symbol BTCUSD --resolution 1h --start 2024-01-01T00:00:00Z --end 2024-01-08T00:00:00Z
python -m scripts.audit_coverage --symbols BTCUSD ETHUSD --resolutions 5m 15m 1h
python -m scripts.record_candle_gaps --symbols BTCUSD ETHUSD --resolutions 5m 15m
python -m scripts.collect_news --once
python -m scripts.stream_market_data
```

The public WebSocket collector subscribes to Delta's current public `ticker`, `trades`, `ob_l1`, `ob_updates`, `mark_price`, `funding_rate`, `spot_price`, and `candlestick_{5m,15m,1h}` channels. Index-price subscriptions are explicitly mapped for BTCUSD (`.DEXBTUSD`) and ETHUSD (`.DEETHUSD`); add a verified mapping before subscribing to another instrument. Order-book updates are sequence-checked; a sequence gap tears down the connection so a new subscription receives a clean snapshot. WebSocket market events are persisted with their raw payload, venue, event timestamp, and receive timestamp. Delta's observed `spot_price` messages may have a null exchange timestamp; those events use local receive time for `event_time`, while the raw payload preserves the null exchange timestamp. Mark price, funding rate, index price, and available ticker open interest are also normalized into `market_metrics`, with source event type and payload retained. Historical candle retrieval chunks ranges below Delta's documented 2,000-candle response maximum and uses idempotent PostgreSQL upserts; each backfill attempt is recorded in `ingest_runs`. The stream service also refreshes completed BTCUSD/ETHUSD 1h candles from Delta REST for the preceding eight days every 15 minutes, repairing recent gaps without replacing the still-forming current-hour WebSocket candle.

## Shared market features

`GET /api/v1/market/features?asset=BTC&resolution=15m` returns a deterministic `shared-market-features-v1` vector in an explicit, stable feature order. Supported assets are BTC and ETH; supported candle resolutions are 5m, 15m, and 1h. The default `as_of` is the current UTC boundary for the selected resolution. An optional historical `as_of` must include a timezone, align exactly to that UTC boundary, and not be in the future. Only completed candles and source observations at or before `as_of` are eligible.

The vector covers price, trend, momentum, volatility, volume, top-of-book/order-flow, derivatives, and BTC/ETH market context. Each feature includes its value and unit, source, source timestamp, window, sample count, and availability status. Insufficient, stale, malformed, or discontinuous inputs are reported with a null value and reason rather than filled with zero. Price features require contiguous completed candles; 15-minute order-flow features require event coverage at both ends of the window. Trade volume is unsigned contract volume: the stored trade-role field is not treated as buy/sell direction. This API is a reusable research data interface, not a forecast, trading signal, or order endpoint.

`GET /api/v1/market/data-continuity` checks both assets at the latest completed 1h boundary and reports per-asset 29-feature completeness, feature source-group coverage, and the provenance/reasons for unavailable features. It also reports latest timestamps, ages, and freshness states for the live order-book, trade, funding, open-interest, mark-price, and index-price feeds. An incomplete report still returns HTTP 200 because it is a diagnostic, not a readiness gate; it does not change `/health/ready` or `/api/v1/system/status`.

## Triple Barrier research labels

`app.market_data.labels.label_triple_barrier` creates a directional label from future OHLC bars without placing orders. Every run must explicitly provide `profit_target_bps`, `stop_loss_bps`, `round_trip_cost_bps`, and `max_holding_bars`; there are deliberately no environment or strategy defaults. The round-trip cost is a combined fee-and-slippage assumption for entry and exit. The gross profit barrier is moved outward by that cost, and the gross stop barrier is moved toward entry so the reported net return reaches the configured stop after cost.

Entry uses the next candle's open at `decision_time`; only subsequent OHLC is inspected. A profit or stop touch labels the event `1` or `-1`, while vertical expiry uses the sign of the after-cost return. If one OHLC candle touches both barriers, intrabar order is unknown and the sample is marked ambiguous. Gaps, invalid OHLC, and incomplete horizons produce an explicit unlabelled status. The result carries the full parameter set and label-end timestamp for reproducible downstream purging; labels are research targets, not a strategy or execution record.

## Building research datasets

Run the dataset builder from the backend repository root after PostgreSQL has been migrated and populated with Delta candles. It reads shared feature vectors and future OHLC from one PostgreSQL read-only `REPEATABLE READ` transaction. Label and split settings are required on every run; the values below are illustrative only and are not calibrated or recommended:

```powershell
python -m scripts.build_dataset `
  --asset BTC ETH `
  --resolution 15m `
  --start 2024-06-01T00:00:00Z `
  --end 2024-07-01T00:00:00Z `
  --profit-target-bps 100 `
  --stop-loss-bps 80 `
  --round-trip-cost-bps 10 `
  --max-holding-bars 8 `
  --train-fraction 0.7 `
  --validation-fraction 0.15 `
  --embargo-bars 8 `
  --output-dir research-datasets
```

The start is inclusive and the end is exclusive; both must be UTC resolution boundaries and the end cannot be in the future. The builder loads the required feature lookback and label horizon beyond the requested decision-time range. It emits only samples whose complete shared feature vector and Triple Barrier outcome are available. Missing features, ambiguous barrier touches, gaps, invalid candles, and incomplete label horizons are written to `excluded.jsonl`, not imputed or guessed.

Samples are split chronologically across shared decision-time boundaries for all selected assets. Labels that overlap the next split are purged, and the explicit embargo is applied after validation and test boundaries. The versioned output directory contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, `excluded.jsonl`, and a `manifest.json` with source lineage, label/split parameters, boundaries, counts, and output checksums. Its content-derived version changes when input features, labels, lineage, or configuration changes; `computed_at` is excluded so unchanged historical inputs do not get a new version merely because the build ran later. Existing outputs are checksum-verified before an identical build is reused.

## Market regime baseline

`GET /api/v1/market/regimes` returns research-only BTC and ETH regime snapshots. It uses completed, contiguous 1h candles (at least 50 required), EMA20/EMA50 spread, 12h directional consistency, 24h return, and 24h realized volatility. A candle older than two hours, an insufficient history, or a gap in the 50-candle window produces `unavailable`, not a directional fallback. The `ohlcv-regime-v1` baseline labels volatility at 5% or higher as `high_volatility`; otherwise a trend requires an EMA spread of at least +/-0.15% and directional consistency of at least 0.60, with remaining valid data labeled `range`. A 24h-vs-prior-6d volume ratio is included when 168 contiguous candles are available.

Latest funding, open interest, mark/index prices, their observation times, and the measured open-interest change window are returned as context. Derivatives do not affect the regime label. Each result includes its evidence, an uncalibrated deterministic confidence heuristic, and invalidation conditions. These thresholds have not been calibrated against an independently labeled evaluation set; outputs are not forecasts, trading signals, or authorization to place orders.

`POST /api/v1/market/bias/assess` assesses BTC and ETH and persists one locked decision per asset per UTC calendar day; `GET /api/v1/market/bias` reads the persisted decisions and returns an empty list until the first assessment. `session-bias-v1` allows `LONG_ONLY` only when the 1h regime, a contiguous 4h EMA6/EMA12 context, and at least three recent model-scored headlines all agree bullish; `SHORT_ONLY` requires the corresponding bearish alignment. Missing, stale, neutral, conflicting, or insufficient data yields `NO_TRADE`. The headline window is 24h; the sentiment mean-confidence floor is 0.55 and weighted-score direction threshold is +/-0.10. Re-assessment never reverses or reactivates a locked direction: if the regime, 4h trend, or news confirmation no longer agrees, an active direction moves only to `suspended` for the rest of that UTC day. Even an initial `NO_TRADE` stays locked for that session. Assessment is stateful and should be invoked explicitly; it is not an order endpoint. No live or paper orders are placed.

`GET /api/v1/market/bias/evaluation?days=30` (7-90 complete UTC days) runs a read-only chronological replay. For consistency, the replay fixes the initial assessment at 00:00 UTC and checks invalidation every four hours through 20:00 UTC; it reports daily decision stability, no-trade reasons, invalidation frequency and post-invalidation returns, and outcome buckets by asset, initial regime, and direction. Decision stability includes `NO_TRADE` labels; directional stability excludes them. It ignores headlines until their publication/observation, first-seen, and model-scoring timestamps are all available to the simulated decision. Only complete hourly candle paths contribute to 24h outcomes. These descriptive results do not establish accuracy or profitability, exclude fees/slippage/fill effects, and do not write session-bias decisions.

## Historical coverage

Delta's `GET /v2/history/candles` returns at most 2,000 candles per response. The backfill helper chunks requests based on the selected resolution and records the venue/source on every candle. It does not fill missing candles or combine other exchanges' data.

`record_candle_gaps` detects missing intervals between stored candles and upserts deterministic `candle_gap` events in `data_quality_events`. Re-running it updates the same event rather than creating duplicate alerts. Events describe gaps in stored Delta history; they do not imply that another source has been used or that a gap has been filled.

## News ingestion

The `news-collector` service polls the public CoinDesk and Cointelegraph RSS feeds every five minutes by default. Override `NEWS_FEED_URLS` with a JSON array of permitted HTTP(S) feed URLs and tune `NEWS_POLL_INTERVAL_SECONDS` or `NEWS_REQUEST_TIMEOUT_SECONDS` in `.env`. Run one collection pass with `python -m scripts.collect_news --once`. The collector stores normalized headlines, canonical links, publication times, and per-feed source identifiers; it deduplicates matching canonical URLs across feeds and does not store article bodies or RSS descriptions. No news API credentials are used.

The `news-sentiment` service classifies unscored headlines locally using the pinned, quantized ONNX export of [`9mark9/finbert-minilm-sentiment`](https://huggingface.co/9mark9/finbert-minilm-sentiment/tree/f1f32e4b7c3aa451bcc3ab1d36e4b7df8680abe2). Its repository declares MIT licensing for the model; the base MiniLM model also declares MIT. The model is downloaded once into the `news_model_cache` volume and then inference runs on CPU without sending headlines to a model API. Predictions store class probabilities, confidence, a BTC/ETH relevance-weighted score, deterministic event category/risk severity, model revision, and scoring time. The model card's benchmark results are unverified and are not treated as a trading-performance claim; sentiment is research data only and does not authorize or trigger orders.

Run a single classification pass on demand with `docker compose exec news-sentiment python -m scripts.classify_news --once`; the service otherwise polls for new headlines every 30 seconds. `GET /api/v1/news/quality?window_hours=24` reports feed success/failure, overall classification coverage, BTC/ETH annotation coverage, event/risk counts, confidence, and ingestion/scoring latency. The report is operational monitoring, not a sentiment-accuracy claim; that requires an independently labeled evaluation set.

The current audit found nine native-history gaps totaling 4,336 missing candles: BTCUSD 5m (3,882), BTCUSD 15m (303), and ETHUSD 5m (151). ETHUSD 15m has no detected gaps. These gaps are recorded as data-quality events and are not filled with another exchange's data. The product catalog observed during initial research lists BTCUSD perpetuals from 2023-12-18 and ETHUSD perpetuals from 2024-02-05. Therefore Delta-native data alone cannot provide three full years for these instruments as of 2026-11-13. A separate, explicitly source-labelled historical source is required if the three-year target is mandatory. Run the coverage audit before model research; do not treat external-source candles as Delta execution data.

## Worker tool boundary

`app.workers.tools` defines role-scoped, validated read-only tool requests for future news, ML, technical, order-flow, validator, and performance workers. Each request must pass `authorize_worker_tool` before a worker adapter dispatches it; unknown tools and role/tool combinations outside the allowlist are rejected, and argument models reject extra fields and invalid ranges. The current tool catalog exposes only reads for the existing market features, regime, session bias, news quality, and bias evaluation interfaces. It has no order-placement, risk mutation, or model-promotion capability.

This module is an authorization policy boundary, not an operating-system sandbox: future worker adapters must route every tool call through it and must not expose raw database, exchange, or unrestricted application access. No workers or tool-dispatch service are implemented yet.

## Supervisor safety core

`app.workers.supervisor` validates worker attempts against an explicitly supplied timeout, retry limit, and output-freshness window. Late, failed, missing-timestamp, stale, and future-dated outputs are rejected; retries are bounded, and exhausted failures suspend the session. Required workers run concurrently through `execute_worker_cycle`. Active sessions suspend when a required role does not provide an accepted result. A suspended session remains suspended until all required outputs are fresh and an explicit resume is requested; stopped sessions are terminal.

`app.workers.persistence` creates BTC/ETH worker sessions and persists append-only initialization, cycle-decision, suspension/recovery, and stop traces. `get_worker_decision_traces` provides a bounded read of the audit trail. Every cycle trace includes each role's attempt count, decision, reason, and start/completion/source timestamps; worker payloads are not stored. PostgreSQL advisory locks prevent overlapping cycles or stop operations for the same session. Apply migration `0007_worker_supervisor` with `alembic upgrade head`. The orchestration is an application service callable by future workflows, not a background scheduler or HTTP endpoint. No actual worker implementations or workflow are registered yet, and every future tool adapter must use the role allowlist above.

## Configuration and infrastructure

`DATABASE_URL`, `REDIS_URL`, `DELTA_REST_BASE_URL`, and `DELTA_PUBLIC_WS_URL` are environment-configurable. The host-run settings defaults use the published PostgreSQL/Redis host ports; Compose overrides them to `postgres:5432` and `redis:6379` internally. The `.env.example` defaults are for Delta India; use the corresponding official global hosts if the deployment/account is on Delta Global. No exchange credentials are required or used.

Run migrations with `alembic upgrade head`. PostgreSQL stores normalized candles plus source-labelled raw market events; Redis stores the latest receive time per asset for ephemeral coordination and freshness monitoring. `/health/ready` and `/api/v1/system/status` return HTTP 503 unless PostgreSQL and Redis are available, the public exchange catalog responds, and fresh WebSocket events have been received for both assets. The liveness endpoint does not depend on these external services.

## Validation

Run the unit tests with:

```powershell
pytest
```

Tests cover exchange pagination, request timestamp units, decimal/UTC normalization, OHLC validation, idempotent PostgreSQL upsert statement generation, live WebSocket candle persistence, deterministic regime thresholds and fail-closed data checks, Triple Barrier labels, chronological dataset splits and versioned exports, current public WebSocket channel names, derivative metric normalization, malformed and out-of-order order-book updates, snapshot/reconnect recovery, missing-candle detection, RSS parsing and deduplication, and sentiment annotation and idempotent persistence. Live Delta, PostgreSQL, Redis, full historical coverage, and container-restart persistence checks require running services and are not represented by these unit tests.
