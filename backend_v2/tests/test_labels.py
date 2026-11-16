from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.market_data.features import FeatureResolution
from app.market_data.labels import (
    BarrierCandle,
    TripleBarrierLabel,
    TripleBarrierParameters,
    label_triple_barrier,
)

DECISION_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
BAR_INTERVALS: dict[FeatureResolution, timedelta] = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
}


def make_bar(
    index: int,
    *,
    open_price: str = "100",
    high: str = "100.2",
    low: str = "99.8",
    close: str = "100",
    resolution: FeatureResolution = "15m",
) -> BarrierCandle:
    return BarrierCandle(
        open_time=DECISION_TIME + BAR_INTERVALS[resolution] * index,
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
    )


def make_parameters() -> TripleBarrierParameters:
    return TripleBarrierParameters(
        profit_target_bps=Decimal("100"),
        stop_loss_bps=Decimal("80"),
        round_trip_cost_bps=Decimal("10"),
        max_holding_bars=3,
    )


def label(
    candles: list[BarrierCandle],
    parameters: TripleBarrierParameters | None = None,
) -> TripleBarrierLabel:
    return label_triple_barrier(
        candles,
        asset="BTC",
        resolution="15m",
        decision_time=DECISION_TIME,
        parameters=parameters or make_parameters(),
    )


def test_profit_and_stop_barriers_include_round_trip_costs() -> None:
    profit_result = label(
        [
            make_bar(0),
            make_bar(
                1,
                open_price="100.2",
                high="101.2",
                low="100.0",
                close="101.0",
            ),
        ]
    )
    stop_result = label(
        [
            make_bar(0),
            make_bar(
                1,
                high="100.2",
                low="99.1",
                close="99.5",
            ),
        ]
    )

    assert profit_result.status == "labeled"
    assert profit_result.barrier == "profit"
    assert profit_result.label == 1
    assert profit_result.parameters == make_parameters()
    assert profit_result.gross_return_bps == pytest.approx(110)
    assert profit_result.net_return_bps == pytest.approx(100)
    assert profit_result.label_end_time == DECISION_TIME + BAR_INTERVALS["15m"] * 2
    assert stop_result.status == "labeled"
    assert stop_result.barrier == "stop"
    assert stop_result.label == -1
    assert stop_result.gross_return_bps == pytest.approx(-70)
    assert stop_result.net_return_bps == pytest.approx(-80)


def test_open_gap_uses_observed_open_and_double_touch_is_ambiguous() -> None:
    gap_result = label(
        [
            make_bar(0),
            make_bar(
                1,
                open_price="101.5",
                high="101.7",
                low="101.4",
                close="101.6",
            ),
        ]
    )
    ambiguous_result = label(
        [
            make_bar(
                0,
                high="101.2",
                low="99.1",
            ),
        ]
    )

    assert gap_result.barrier == "profit"
    assert gap_result.exit_price == pytest.approx(101.5)
    assert gap_result.net_return_bps == pytest.approx(140)
    assert ambiguous_result.status == "ambiguous"
    assert ambiguous_result.label is None
    assert ambiguous_result.barrier == "ambiguous"


def test_vertical_barrier_uses_after_cost_return_and_requires_full_horizon() -> None:
    vertical = label([make_bar(0), make_bar(1), make_bar(2, close="100.02")])
    incomplete = label([make_bar(0), make_bar(1)])

    assert vertical.status == "labeled"
    assert vertical.barrier == "vertical"
    assert vertical.net_return_bps == pytest.approx(-8)
    assert vertical.label == -1
    assert incomplete.status == "insufficient_data"
    assert incomplete.label is None
    assert incomplete.label_end_time == DECISION_TIME + BAR_INTERVALS["15m"] * 2


def test_gaps_invalid_ohlc_and_bad_parameters_fail_closed() -> None:
    gap = label(
        [make_bar(0), make_bar(2), make_bar(3)]
    )
    invalid = label(
        [
            make_bar(0),
            make_bar(1, high="99.9", low="99.8", close="100"),
            make_bar(2),
        ]
    )
    assert gap.status == "gap"
    assert gap.label is None
    assert invalid.status == "invalid_data"
    assert invalid.label is None
    with pytest.raises(ValidationError, match="stop_loss_bps"):
        TripleBarrierParameters(
            profit_target_bps=Decimal("100"),
            stop_loss_bps=Decimal("10"),
            round_trip_cost_bps=Decimal("10"),
            max_holding_bars=3,
        )
    with pytest.raises(ValidationError, match="10,000"):
        TripleBarrierParameters(
            profit_target_bps=Decimal("100"),
            stop_loss_bps=Decimal("10000"),
            round_trip_cost_bps=Decimal("0"),
            max_holding_bars=3,
        )
    with pytest.raises(ValueError, match="timezone"):
        label_triple_barrier(
            [make_bar(0)],
            asset="BTC",
            resolution="15m",
            decision_time=DECISION_TIME.replace(tzinfo=None),
            parameters=make_parameters(),
        )
