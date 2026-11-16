"""Worker 4: Technical Analysis Agent for DeltaBot 2.0.

Performs multi-timeframe analysis (1h context, 15m confirmation, 5m entry), detects technical setups
(EMA pullbacks, Breakout/Retest, VWAP Reclaim, Momentum continuation), and calculates stop/target levels.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from typing import Literal

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.market_data.regimes import ASSET_SYMBOLS, Asset
from app.models import Candle
from app.workers.supervisor import WorkerOutput

SetupType = Literal[
    "EMA_PULLBACK",
    "BREAKOUT_RETEST",
    "VWAP_RECLAIM",
    "MOMENTUM_CONTINUATION",
    "NONE",
]
TradeDirection = Literal["LONG", "SHORT", "NEUTRAL"]


@dataclass(frozen=True)
class CandleBar:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return [float("nan")] * len(prices)
    alpha = 2.0 / (period + 1.0)
    ema = [float("nan")] * len(prices)
    # Seed with SMA
    ema[period - 1] = float(np.mean(prices[:period]))
    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calculate Relative Strength Index."""
    if len(prices) <= period:
        return [float("nan")] * len(prices)
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    rsi = [float("nan")] * len(prices)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0.0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_vwap(candles: list[CandleBar]) -> float | None:
    """Calculate Volume-Weighted Average Price from candle bars."""
    if not candles:
        return None
    cum_vol = 0.0
    cum_vol_price = 0.0
    for c in candles:
        typical_price = (c.high + c.low + c.close) / 3.0
        cum_vol_price += typical_price * c.volume
        cum_vol += c.volume
    if cum_vol <= 0.0:
        return None
    return cum_vol_price / cum_vol


class TechnicalAnalysisAgent:
    """Specialist worker executing multi-timeframe technical pattern recognition."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def _fetch_candles(
        self,
        connection: AsyncConnection,
        symbol: str,
        resolution: str,
        as_of: datetime,
        limit: int = 60,
    ) -> list[CandleBar]:
        query = (
            select(
                Candle.open_time,
                Candle.open,
                Candle.high,
                Candle.low,
                Candle.close,
                Candle.volume,
            )
            .where(
                Candle.venue == "delta",
                Candle.symbol == symbol,
                Candle.resolution == resolution,
                Candle.open_time < as_of,
            )
            .order_by(Candle.open_time.desc())
            .limit(limit)
        )
        res = await connection.execute(query)
        rows = res.all()
        # Sort chronologically
        rows_sorted = sorted(rows, key=lambda r: r[0])
        return [
            CandleBar(
                open_time=r[0],
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
            )
            for r in rows_sorted
        ]

    async def analyze(
        self,
        asset: Asset,
        as_of: datetime,
        *,
        connection: AsyncConnection | None = None,
    ) -> WorkerOutput:
        symbol = ASSET_SYMBOLS[asset][0]

        async def _run(conn: AsyncConnection) -> WorkerOutput:
            # Fetch 1h context (higher timeframe trend)
            candles_1h = await self._fetch_candles(conn, symbol, "1h", as_of, limit=60)
            # Fetch 15m confirmation
            candles_15m = await self._fetch_candles(conn, symbol, "15m", as_of, limit=60)

            if len(candles_1h) < 30 or len(candles_15m) < 30:
                return WorkerOutput(
                    output_as_of=as_of,
                    payload={
                        "asset": asset,
                        "setup_detected": False,
                        "setup_type": "NONE",
                        "direction": "NEUTRAL",
                        "confidence": 0.0,
                        "entry_reference_price": None,
                        "stop_reference_price": None,
                        "target_reference_price": None,
                        "reason": "Insufficient candle history",
                    },
                )

            closes_1h = [c.close for c in candles_1h]
            ema_21_1h = calculate_ema(closes_1h, 21)[-1]
            ema_50_1h = calculate_ema(closes_1h, 50)[-1] if len(closes_1h) >= 50 else float("nan")

            closes_15m = [c.close for c in candles_15m]
            ema_8_15m = calculate_ema(closes_15m, 8)[-1]
            ema_21_15m = calculate_ema(closes_15m, 21)[-1]
            rsi_15m = calculate_rsi(closes_15m, 14)[-1]
            vwap_15m = calculate_vwap(candles_15m[-24:]) or closes_15m[-1]

            current_price = closes_15m[-1]

            # 1. Higher-timeframe bias context
            htf_bullish = closes_1h[-1] > ema_21_1h and (math.isnan(ema_50_1h) or ema_21_1h > ema_50_1h)
            htf_bearish = closes_1h[-1] < ema_21_1h and (math.isnan(ema_50_1h) or ema_21_1h < ema_50_1h)

            setup_detected = False
            setup_type: SetupType = "NONE"
            direction: TradeDirection = "NEUTRAL"
            confidence = 0.0
            stop_price = None
            target_price = None

            # Recent support / resistance from last 20 15m candles
            recent_low = min(c.low for c in candles_15m[-20:])
            recent_high = max(c.high for c in candles_15m[-20:])

            # Setup 1: EMA Pullback
            if htf_bullish and ema_8_15m > ema_21_15m:
                # Price pulled back near 21 EMA and RSI between 40 and 65
                if abs(current_price - ema_21_15m) / current_price < 0.004 and 40.0 <= rsi_15m <= 65.0:
                    setup_detected = True
                    setup_type = "EMA_PULLBACK"
                    direction = "LONG"
                    confidence = 0.65
                    stop_price = round(min(recent_low, current_price * 0.992), 2)
                    target_price = round(current_price * 1.012, 2)
            elif htf_bearish and ema_8_15m < ema_21_15m:
                if abs(current_price - ema_21_15m) / current_price < 0.004 and 35.0 <= rsi_15m <= 60.0:
                    setup_detected = True
                    setup_type = "EMA_PULLBACK"
                    direction = "SHORT"
                    confidence = 0.65
                    stop_price = round(max(recent_high, current_price * 1.008), 2)
                    target_price = round(current_price * 0.988, 2)

            # Setup 2: VWAP Reclaim / Rejection
            if not setup_detected:
                if htf_bullish and current_price >= vwap_15m and candles_15m[-2].close < vwap_15m:
                    setup_detected = True
                    setup_type = "VWAP_RECLAIM"
                    direction = "LONG"
                    confidence = 0.60
                    stop_price = round(vwap_15m * 0.995, 2)
                    target_price = round(current_price * 1.010, 2)
                elif htf_bearish and current_price <= vwap_15m and candles_15m[-2].close > vwap_15m:
                    setup_detected = True
                    setup_type = "VWAP_RECLAIM"
                    direction = "SHORT"
                    confidence = 0.60
                    stop_price = round(vwap_15m * 1.005, 2)
                    target_price = round(current_price * 0.990, 2)

            # Setup 3: Momentum Continuation
            if not setup_detected:
                if htf_bullish and rsi_15m > 58.0 and current_price > ema_8_15m > ema_21_15m:
                    setup_detected = True
                    setup_type = "MOMENTUM_CONTINUATION"
                    direction = "LONG"
                    confidence = 0.58
                    stop_price = round(ema_21_15m * 0.995, 2)
                    target_price = round(current_price * 1.012, 2)
                elif htf_bearish and rsi_15m < 42.0 and current_price < ema_8_15m < ema_21_15m:
                    setup_detected = True
                    setup_type = "MOMENTUM_CONTINUATION"
                    direction = "SHORT"
                    confidence = 0.58
                    stop_price = round(ema_21_15m * 1.005, 2)
                    target_price = round(current_price * 0.988, 2)

            return WorkerOutput(
                output_as_of=as_of,
                payload={
                    "asset": asset,
                    "setup_detected": setup_detected,
                    "setup_type": setup_type,
                    "direction": direction,
                    "confidence": confidence,
                    "entry_reference_price": current_price if setup_detected else None,
                    "stop_reference_price": stop_price,
                    "target_reference_price": target_price,
                    "indicators": {
                        "price": current_price,
                        "ema_21_1h": ema_21_1h,
                        "ema_8_15m": ema_8_15m,
                        "ema_21_15m": ema_21_15m,
                        "rsi_15m": rsi_15m,
                        "vwap_15m": vwap_15m,
                    },
                },
            )

        if connection is not None:
            return await _run(connection)
        async with self.engine.connect() as conn:
            return await _run(conn)
