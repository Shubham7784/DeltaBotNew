import argparse
import asyncio
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.database import engine
from app.market_data.backfill import RESOLUTION_SECONDS
from app.market_data.datasets import (
    DatasetSample,
    DatasetSplitParameters,
    write_versioned_dataset,
)
from app.market_data.features import (
    CANDLE_FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    MAX_CANDLE_HISTORY,
    FeatureResolution,
    get_market_feature_vector,
)
from app.market_data.labels import (
    TRIPLE_BARRIER_VERSION,
    BarrierCandle,
    TripleBarrierParameters,
    label_triple_barrier,
)
from app.market_data.regimes import ASSET_SYMBOLS, Asset
from app.models import Candle


@dataclass(frozen=True)
class StoredCandle:
    symbol: str
    candle: BarrierCandle
    source: str


def _parse_asset(value: str) -> Asset:
    if value == "BTC":
        return "BTC"
    if value == "ETH":
        return "ETH"
    raise argparse.ArgumentTypeError("Asset must be BTC or ETH")


def _parse_resolution(value: str) -> FeatureResolution:
    if value in ("5m", "15m", "1h"):
        return value
    raise argparse.ArgumentTypeError("Resolution must be 5m, 15m, or 1h")


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Timestamp must be ISO 8601, for example 2026-01-01T00:00:00Z"
        ) from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("Value must be a decimal number") from error
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("Value must be finite")
    return parsed


def _validate_time_range(
    *,
    start: datetime,
    end: datetime,
    resolution: FeatureResolution,
    now: datetime,
) -> None:
    if start.tzinfo is None or end.tzinfo is None or now.tzinfo is None:
        raise ValueError("Dataset bounds and current time must include a timezone")
    if end <= start:
        raise ValueError("Dataset end must be after start")
    if end > now:
        raise ValueError("Dataset end cannot be in the future")
    interval = RESOLUTION_SECONDS[resolution]
    if int(start.timestamp()) % interval or int(end.timestamp()) % interval:
        raise ValueError("Dataset bounds must align to the resolution's UTC boundary")


async def generate_dataset(
    *,
    assets: list[Asset],
    resolution: FeatureResolution,
    start: datetime,
    end: datetime,
    label_parameters: TripleBarrierParameters,
    split_parameters: DatasetSplitParameters,
    output_dir: Path,
    connection: AsyncConnection,
    feature_schema: str = CANDLE_FEATURE_SCHEMA_VERSION,
) -> tuple[Path, dict[str, object]]:
    run_started_at = datetime.now(UTC)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Dataset bounds must include a timezone")
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)
    if len(set(assets)) != len(assets):
        raise ValueError("Dataset asset list must not contain duplicates")
    if not assets:
        raise ValueError("At least one asset must be selected")
    assets = sorted(assets)
    _validate_time_range(
        start=start,
        end=end,
        resolution=resolution,
        now=run_started_at,
    )
    interval = timedelta(seconds=RESOLUTION_SECONDS[resolution])
    symbols = [ASSET_SYMBOLS[asset][0] for asset in assets]
    symbol_to_asset = {
        ASSET_SYMBOLS[asset][0]: asset for asset in assets
    }
    candle_query_start = start - interval * MAX_CANDLE_HISTORY
    candle_query_end = end + interval * label_parameters.max_holding_bars
    result = await connection.execute(
        select(
            Candle.symbol,
            Candle.open_time,
            Candle.open,
            Candle.high,
            Candle.low,
            Candle.close,
            Candle.source,
        )
        .where(
            Candle.venue == "delta",
            Candle.symbol.in_(symbols),
            Candle.resolution == resolution,
            Candle.open_time >= candle_query_start,
            Candle.open_time < candle_query_end,
        )
        .order_by(Candle.symbol, Candle.open_time)
    )
    stored_candles = [
        StoredCandle(
            symbol=row[0],
            candle=BarrierCandle(
                open_time=row[1],
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
            ),
            source=row[6],
        )
        for row in result.all()
    ]

    candles_by_asset: dict[Asset, list[StoredCandle]] = {
        asset: [] for asset in assets
    }
    for stored_candle in stored_candles:
        candles_by_asset[symbol_to_asset[stored_candle.symbol]].append(stored_candle)
    decision_times_by_asset: dict[Asset, list[datetime]] = {}
    for asset, asset_candles in candles_by_asset.items():
        decision_times_by_asset[asset] = sorted(
            {
                stored.candle.open_time.astimezone(UTC) + interval
                for stored in asset_candles
                if start
                <= stored.candle.open_time.astimezone(UTC) + interval
                < end
            }
        )

    complete_samples: list[DatasetSample] = []
    generation_exclusions: list[dict[str, object]] = []
    for asset in assets:
        asset_candles = candles_by_asset[asset]
        candle_open_times = [
            stored.candle.open_time.astimezone(UTC) for stored in asset_candles
        ]
        for decision_time in decision_times_by_asset[asset]:
            entry_index = bisect_left(candle_open_times, decision_time)
            future_candles = [
                stored.candle for stored in asset_candles[entry_index:]
            ]
            label = label_triple_barrier(
                future_candles,
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                parameters=label_parameters,
            )
            if label.status != "labeled":
                generation_exclusions.append(
                    {
                        "asset": asset,
                        "decision_time": decision_time.isoformat(),
                        "reason": f"label_{label.status}",
                        "details": label.reason,
                        "label_end_time": (
                            label.label_end_time.isoformat()
                            if label.label_end_time
                            else None
                        ),
                    }
                )
                continue

            feature_vector = await get_market_feature_vector(
                engine,
                asset=asset,
                resolution=resolution,
                as_of=decision_time,
                now=run_started_at,
                connection=connection,
                schema_version=feature_schema,
            )
            unavailable = [
                {
                    "name": feature.name,
                    "status": feature.status,
                    "reason": feature.missing_reason,
                }
                for feature in feature_vector.features
                if feature.status != "available"
            ]
            if unavailable:
                generation_exclusions.append(
                    {
                        "asset": asset,
                        "decision_time": decision_time.isoformat(),
                        "reason": "incomplete_feature_vector",
                        "details": unavailable,
                        "label_end_time": label.label_end_time.isoformat()
                        if label.label_end_time
                        else None,
                    }
                )
                continue

            source_start = decision_time - interval * MAX_CANDLE_HISTORY
            source_end = label.label_end_time or decision_time
            source_start_index = bisect_left(candle_open_times, source_start)
            source_end_index = bisect_left(candle_open_times, source_end)
            candle_sources = tuple(
                sorted(
                    {
                        stored.source
                        for stored in asset_candles[
                            source_start_index:source_end_index
                        ]
                    }
                )
            )
            complete_samples.append(
                DatasetSample(
                    features=feature_vector,
                    label=label,
                    candle_sources=candle_sources,
                )
            )

    source_counts: dict[str, dict[str, int]] = {}
    for symbol in symbols:
        source_counts[symbol] = dict(
            sorted(
                Counter(
                    stored.source
                    for stored in stored_candles
                    if stored.symbol == symbol
                ).items()
            )
        )
    source_lineage: dict[str, object] = {
        "venue": "delta",
        "assets": list(assets),
        "symbols": {asset: ASSET_SYMBOLS[asset][0] for asset in assets},
        "resolution": resolution,
        "requested_start": start.isoformat(),
        "requested_end_exclusive": end.isoformat(),
        "candle_query_start": candle_query_start.isoformat(),
        "candle_query_end_exclusive": candle_query_end.isoformat(),
        "candle_rows_loaded": len(stored_candles),
        "candle_sources_by_symbol": source_counts,
        "feature_schema_version": feature_schema,
        "label_algorithm_version": TRIPLE_BARRIER_VERSION,
    }
    return write_versioned_dataset(
        output_dir=output_dir,
        samples=complete_samples,
        generation_exclusions=generation_exclusions,
        source_lineage=source_lineage,
        split_parameters=split_parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a versioned, leakage-controlled BTC/ETH research dataset"
    )
    parser.add_argument(
        "--asset",
        nargs="+",
        type=_parse_asset,
        default=["BTC", "ETH"],
    )
    parser.add_argument(
        "--resolution",
        required=True,
        type=_parse_resolution,
    )
    parser.add_argument("--start", required=True, type=_parse_datetime)
    parser.add_argument("--end", required=True, type=_parse_datetime)
    parser.add_argument(
        "--feature-schema",
        default=CANDLE_FEATURE_SCHEMA_VERSION,
        choices=[CANDLE_FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION],
        help="Feature schema version to extract",
    )
    parser.add_argument("--profit-target-bps", required=True, type=_parse_decimal)
    parser.add_argument("--stop-loss-bps", required=True, type=_parse_decimal)
    parser.add_argument("--round-trip-cost-bps", required=True, type=_parse_decimal)
    parser.add_argument("--max-holding-bars", required=True, type=int)
    parser.add_argument("--train-fraction", required=True, type=float)
    parser.add_argument("--validation-fraction", required=True, type=float)
    parser.add_argument("--embargo-bars", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    label_parameters = TripleBarrierParameters(
        profit_target_bps=args.profit_target_bps,
        stop_loss_bps=args.stop_loss_bps,
        round_trip_cost_bps=args.round_trip_cost_bps,
        max_holding_bars=args.max_holding_bars,
    )
    split_parameters = DatasetSplitParameters(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        embargo_bars=args.embargo_bars,
    )

    async def generate_and_dispose() -> tuple[Path, dict[str, object]]:
        try:
            async with engine.connect() as database_connection:
                connection = await database_connection.execution_options(
                    isolation_level="REPEATABLE READ"
                )
                async with connection.begin():
                    await connection.execute(text("SET TRANSACTION READ ONLY"))
                    return await generate_dataset(
                        assets=args.asset,
                        resolution=args.resolution,
                        start=args.start,
                        end=args.end,
                        label_parameters=label_parameters,
                        split_parameters=split_parameters,
                        output_dir=args.output_dir,
                        connection=connection,
                        feature_schema=args.feature_schema,
                    )
        finally:
            await engine.dispose()

    output_path, manifest = asyncio.run(generate_and_dispose())
    print(f"Dataset: {output_path}")
    print(f"Version: {manifest['dataset_version']}")
    print(f"Counts: {manifest['sample_counts']}")


if __name__ == "__main__":
    main()
