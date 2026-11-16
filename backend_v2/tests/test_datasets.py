from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.market_data.datasets import (
    DatasetSample,
    DatasetSplitParameters,
    split_chronologically,
    write_versioned_dataset,
)
from app.market_data.features import (
    FEATURE_DEFINITIONS,
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    FeatureValue,
    MarketFeatureVector,
)
from app.market_data.labels import (
    TRIPLE_BARRIER_VERSION,
    TripleBarrierLabel,
    TripleBarrierParameters,
)
from app.market_data.regimes import ASSET_SYMBOLS, Asset

START = datetime(2026, 1, 1, tzinfo=UTC)
INTERVAL = timedelta(minutes=15)
LABEL_PARAMETERS = TripleBarrierParameters(
    profit_target_bps=Decimal("100"),
    stop_loss_bps=Decimal("80"),
    round_trip_cost_bps=Decimal("10"),
    max_holding_bars=2,
)


def make_sample(
    asset: Asset,
    index: int,
    *,
    computed_at_offset: timedelta = timedelta(minutes=1),
) -> DatasetSample:
    decision_time = START + INTERVAL * index
    feature_values = [
        FeatureValue(
            name=definition.name,
            family=definition.family,
            value=1.0,
            unit=definition.unit,
            source=definition.source,
            source_as_of=decision_time,
            window_start=decision_time - INTERVAL,
            window_end=decision_time,
            sample_count=1,
            status="available",
            missing_reason=None,
        )
        for definition in FEATURE_DEFINITIONS
    ]
    feature_vector = MarketFeatureVector(
        schema_version=FEATURE_SCHEMA_VERSION,
        asset=asset,
        symbol=ASSET_SYMBOLS[asset][0],
        resolution="15m",
        as_of=decision_time,
        computed_at=decision_time + computed_at_offset,
        feature_order=list(FEATURE_ORDER),
        features=feature_values,
        interpretation="Test data",
    )
    label = TripleBarrierLabel(
        algorithm_version=TRIPLE_BARRIER_VERSION,
        asset=asset,
        resolution="15m",
        parameters=LABEL_PARAMETERS,
        decision_time=decision_time,
        entry_time=decision_time,
        label_end_time=decision_time + INTERVAL * 2,
        entry_price=100,
        exit_price=101,
        holding_bars=2,
        gross_return_bps=100,
        net_return_bps=90,
        round_trip_cost_bps=10,
        label=1,
        barrier="profit",
        status="labeled",
        reason=None,
    )
    return DatasetSample(
        features=feature_vector,
        label=label,
        candle_sources=("delta_rest", "delta_ws"),
    )


def test_splits_are_chronological_purged_and_embargoed_across_assets() -> None:
    samples = [
        make_sample(asset, index)
        for index in range(20)
        for asset in ("BTC", "ETH")
    ]
    splits = split_chronologically(
        samples,
        resolution="15m",
        parameters=DatasetSplitParameters(
            train_fraction=0.5,
            validation_fraction=0.25,
            embargo_bars=1,
        ),
    )

    assert splits.validation_start == START + INTERVAL * 10
    assert splits.test_start == START + INTERVAL * 15
    assert all(
        sample.label.label_end_time < splits.validation_start
        for sample in splits.train
    )
    assert all(
        sample.features.as_of >= splits.validation_start + INTERVAL
        and sample.label.label_end_time < splits.test_start
        for sample in splits.validation
    )
    assert all(
        sample.features.as_of >= splits.test_start + INTERVAL
        for sample in splits.test
    )
    assert len(splits.train) == 16
    assert len(splits.validation) == 4
    assert len(splits.test) == 8
    assert {item.reason for item in splits.excluded} == {
        "purged_train_validation_overlap",
        "purged_validation_test_overlap",
        "validation_embargo",
        "test_embargo",
    }


def test_dataset_export_is_content_versioned_and_ignores_build_time(tmp_path) -> None:
    samples = [make_sample("BTC", index) for index in range(20)]
    split_parameters = DatasetSplitParameters(
        train_fraction=0.5,
        validation_fraction=0.25,
        embargo_bars=0,
    )
    source_lineage = {
        "venue": "delta",
        "asset": "BTC",
        "resolution": "15m",
        "range_start": START.isoformat(),
    }
    output_path, manifest = write_versioned_dataset(
        output_dir=tmp_path,
        samples=samples,
        generation_exclusions=[],
        source_lineage=source_lineage,
        split_parameters=split_parameters,
    )
    rebuilt_path, rebuilt_manifest = write_versioned_dataset(
        output_dir=tmp_path,
        samples=[
            make_sample("BTC", index, computed_at_offset=timedelta(days=1))
            for index in range(20)
        ],
        generation_exclusions=[],
        source_lineage=source_lineage,
        split_parameters=split_parameters,
    )

    assert rebuilt_path == output_path
    assert rebuilt_manifest["dataset_version"] == manifest["dataset_version"]
    assert (output_path / "manifest.json").is_file()
    assert (output_path / "train.jsonl").is_file()
    assert (output_path / "validation.jsonl").is_file()
    assert (output_path / "test.jsonl").is_file()
    assert (output_path / "excluded.jsonl").is_file()

    (output_path / "train.jsonl").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum validation"):
        write_versioned_dataset(
            output_dir=tmp_path,
            samples=samples,
            generation_exclusions=[],
            source_lineage=source_lineage,
            split_parameters=split_parameters,
        )


def test_empty_dataset_error_reports_exclusion_reasons(tmp_path) -> None:
    with pytest.raises(ValueError, match="incomplete_feature_vector"):
        write_versioned_dataset(
            output_dir=tmp_path,
            samples=[],
            generation_exclusions=[
                {
                    "asset": "BTC",
                    "decision_time": START.isoformat(),
                    "reason": "incomplete_feature_vector",
                }
            ],
            source_lineage={"venue": "delta"},
            split_parameters=DatasetSplitParameters(
                train_fraction=0.7,
                validation_fraction=0.15,
                embargo_bars=8,
            ),
        )


def test_split_error_reports_distinct_decision_time_count() -> None:
    samples = [
        make_sample(asset, index)
        for index in range(2)
        for asset in ("BTC", "ETH")
    ]

    with pytest.raises(ValueError, match="found 2"):
        split_chronologically(
            samples,
            resolution="15m",
            parameters=DatasetSplitParameters(
                train_fraction=0.7,
                validation_fraction=0.15,
                embargo_bars=8,
            ),
        )
