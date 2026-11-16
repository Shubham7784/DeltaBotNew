"""Worker 6: Order Flow Agent for DeltaBot 2.0.

Analyzes top-of-book depth, bid/ask size imbalance, spread, and aggressive volume delta.
If data is stale or missing, explicitly fails closed with status 'UNAVAILABLE'.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.market_data.regimes import ASSET_SYMBOLS, Asset
from app.models import MarketEvent
from app.workers.supervisor import WorkerOutput

OrderFlowBias = Literal["BULLISH", "BEARISH", "NEUTRAL", "UNAVAILABLE"]
OrderFlowStatus = Literal["AVAILABLE", "UNAVAILABLE"]


class OrderFlowAgent:
    """Specialist worker executing real-time order-book and aggressive flow analysis."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def analyze(
        self,
        asset: Asset,
        as_of: datetime,
        *,
        max_age_seconds: int = 120,
        connection: AsyncConnection | None = None,
    ) -> WorkerOutput:
        symbol = ASSET_SYMBOLS[asset][0]
        window_start = as_of - timedelta(seconds=max_age_seconds)

        async def _run(conn: AsyncConnection) -> WorkerOutput:
            # 1. Fetch latest ob_l1 event
            ob_query = (
                select(MarketEvent.event_time, MarketEvent.payload)
                .where(
                    MarketEvent.venue == "delta",
                    MarketEvent.symbol == symbol,
                    MarketEvent.event_type == "ob_l1",
                    MarketEvent.event_time <= as_of,
                    MarketEvent.event_time >= window_start,
                )
                .order_by(MarketEvent.event_time.desc())
                .limit(1)
            )
            ob_res = await conn.execute(ob_query)
            ob_row = ob_res.first()

            if ob_row is None:
                return WorkerOutput(
                    output_as_of=as_of,
                    payload={
                        "asset": asset,
                        "status": "UNAVAILABLE",
                        "order_flow_bias": "UNAVAILABLE",
                        "imbalance": None,
                        "spread_bps": None,
                        "trade_count_15m": 0,
                        "volume_delta_contracts": None,
                        "reason": f"No recent ob_l1 event within {max_age_seconds}s",
                    },
                )

            ob_time, ob_payload = ob_row
            best_bid_size = float(ob_payload.get("best_bid_size") or 0.0)
            best_ask_size = float(ob_payload.get("best_ask_size") or 0.0)
            best_bid_price = float(ob_payload.get("best_bid_price") or 0.0)
            best_ask_price = float(ob_payload.get("best_ask_price") or 0.0)

            # Imbalance
            total_size = best_bid_size + best_ask_size
            imbalance = (best_bid_size - best_ask_size) / total_size if total_size > 0 else 0.0

            # Spread bps
            mid = (best_bid_price + best_ask_price) / 2.0
            spread_bps = ((best_ask_price - best_bid_price) / mid * 10000.0) if mid > 0 else 0.0

            # 2. Fetch trade events over 15m window
            trade_window_start = as_of - timedelta(minutes=15)
            trades_query = (
                select(MarketEvent.payload)
                .where(
                    MarketEvent.venue == "delta",
                    MarketEvent.symbol == symbol,
                    MarketEvent.event_type == "trades",
                    MarketEvent.event_time <= as_of,
                    MarketEvent.event_time >= trade_window_start,
                )
            )
            trades_res = await conn.execute(trades_query)
            trades_rows = trades_res.all()

            buy_vol = 0.0
            sell_vol = 0.0
            trade_count = len(trades_rows)

            for t_row in trades_rows:
                p = t_row[0]
                size = float(p.get("size") or 0.0)
                side = str(p.get("side", "")).lower()
                buyer_role = str(p.get("buyer_role", "")).lower()
                # Delta trade attribution: taker is the aggressive side
                if buyer_role == "taker" or side == "buy":
                    buy_vol += size
                else:
                    sell_vol += size

            volume_delta = buy_vol - sell_vol

            # Determine order flow bias
            if imbalance > 0.15 and volume_delta >= 0:
                flow_bias: OrderFlowBias = "BULLISH"
            elif imbalance < -0.15 and volume_delta <= 0:
                flow_bias = "BEARISH"
            else:
                flow_bias = "NEUTRAL"

            return WorkerOutput(
                output_as_of=as_of,
                payload={
                    "asset": asset,
                    "status": "AVAILABLE",
                    "order_flow_bias": flow_bias,
                    "imbalance": round(imbalance, 4),
                    "spread_bps": round(spread_bps, 2),
                    "trade_count_15m": trade_count,
                    "volume_delta_contracts": round(volume_delta, 2),
                    "best_bid": best_bid_price,
                    "best_ask": best_ask_price,
                },
            )

        if connection is not None:
            return await _run(connection)
        async with self.engine.connect() as conn:
            return await _run(conn)
