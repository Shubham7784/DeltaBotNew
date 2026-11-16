import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from app.market_data.backfill import RESOLUTION_SECONDS
from app.market_data.features import FEATURE_SCHEMA_VERSION, MarketFeatureVector
from app.market_data.labels import TRIPLE_BARRIER_VERSION, TripleBarrierLabel

DATASET_FORMAT_VERSION = "chronological-market-dataset-v1"


class DatasetSplitParameters(BaseModel):
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    embargo_bars: int = Field(ge=0)

    @field_validator("train_fraction", "validation_fraction")
    @classmethod
    def fractions_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Split fractions must be finite")
        return value

    @model_validator(mode="after")
    def fractions_must_leave_a_test_split(self) -> "DatasetSplitParameters":
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError(
                "Train and validation fractions must leave samples for testing"
            )
        return self


@dataclass(frozen=True)
class DatasetSample:
    features: MarketFeatureVector
    label: TripleBarrierLabel
    candle_sources: tuple[str, ...]


@dataclass(frozen=True)
class ExcludedDatasetSample:
    sample: DatasetSample
    reason: str


@dataclass(frozen=True)
class ChronologicalSplits:
    train: list[DatasetSample]
    validation: list[DatasetSample]
    test: list[DatasetSample]
    excluded: list[ExcludedDatasetSample]
    validation_start: datetime
    test_start: datetime


def _sample_key(sample: DatasetSample) -> tuple[datetime, str]:
    return sample.features.as_of.astimezone(UTC), sample.features.asset


def _validate_sample(sample: DatasetSample) -> None:
    if sample.features.asset != sample.label.asset:
        raise ValueError("Feature and label assets must match")
    if sample.features.resolution != sample.label.resolution:
        raise ValueError("Feature and label resolutions must match")
    if sample.features.as_of != sample.label.decision_time:
        raise ValueError("Feature as_of and label decision_time must match")
    if sample.label.status != "labeled" or sample.label.label_end_time is None:
        raise ValueError("Only completed labels can enter dataset splits")
    if sample.label.label_end_time < sample.label.decision_time:
        raise ValueError("Label end time cannot precede its decision time")
    if any(feature.status != "available" for feature in sample.features.features):
        raise ValueError("Dataset samples must not contain unavailable features")


def split_chronologically(
    samples: list[DatasetSample],
    *,
    resolution: str,
    parameters: DatasetSplitParameters,
) -> ChronologicalSplits:
    if not samples:
        raise ValueError("At least one labeled sample is required")
    ordered_samples = sorted(samples, key=_sample_key)
    seen_samples: set[tuple[datetime, str]] = set()
    for sample in ordered_samples:
        _validate_sample(sample)
        key = _sample_key(sample)
        if key in seen_samples:
            raise ValueError(f"Duplicate dataset sample: {key}")
        seen_samples.add(key)

    decision_times = sorted({sample.features.as_of.astimezone(UTC) for sample in samples})
    if len(decision_times) < 3:
        raise ValueError(
            "At least three distinct decision times are required; "
            f"found {len(decision_times)}"
        )
    train_cut = int(len(decision_times) * parameters.train_fraction)
    validation_cut = int(
        len(decision_times)
        * (parameters.train_fraction + parameters.validation_fraction)
    )
    if train_cut < 1 or validation_cut <= train_cut or validation_cut >= len(decision_times):
        raise ValueError("Split fractions leave an empty chronological partition")
    validation_start = decision_times[train_cut]
    test_start = decision_times[validation_cut]
    interval = timedelta(seconds=RESOLUTION_SECONDS[resolution])
    validation_embargo_end = validation_start + interval * parameters.embargo_bars
    test_embargo_end = test_start + interval * parameters.embargo_bars

    train: list[DatasetSample] = []
    validation: list[DatasetSample] = []
    test: list[DatasetSample] = []
    excluded: list[ExcludedDatasetSample] = []
    for sample in ordered_samples:
        decision_time = sample.features.as_of.astimezone(UTC)
        label_end = sample.label.label_end_time
        if label_end is None:
            raise ValueError("Completed sample has no label_end_time")
        label_end = label_end.astimezone(UTC)
        if decision_time < validation_start:
            if label_end >= validation_start:
                excluded.append(
                    ExcludedDatasetSample(sample, "purged_train_validation_overlap")
                )
            else:
                train.append(sample)
        elif decision_time < test_start:
            if label_end >= test_start:
                excluded.append(
                    ExcludedDatasetSample(sample, "purged_validation_test_overlap")
                )
            elif decision_time < validation_embargo_end:
                excluded.append(ExcludedDatasetSample(sample, "validation_embargo"))
            else:
                validation.append(sample)
        elif decision_time < test_embargo_end:
            excluded.append(ExcludedDatasetSample(sample, "test_embargo"))
        else:
            test.append(sample)

    if not train or not validation or not test:
        raise ValueError(
            "Purging and embargo leave an empty partition; use a larger time range "
            "or adjust the explicit split parameters."
        )
    return ChronologicalSplits(
        train=train,
        validation=validation,
        test=test,
        excluded=excluded,
        validation_start=validation_start,
        test_start=test_start,
    )


def _sample_payload(sample: DatasetSample) -> dict[str, object]:
    return {
        "asset": sample.features.asset,
        "resolution": sample.features.resolution,
        "decision_time": sample.features.as_of.isoformat(),
        "feature_vector": sample.features.model_dump(
            mode="json",
            exclude={"computed_at"},
        ),
        "label": sample.label.model_dump(mode="json"),
        "candle_sources": list(sample.candle_sources),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def write_versioned_dataset(
    *,
    output_dir: Path,
    samples: list[DatasetSample],
    generation_exclusions: list[dict[str, object]],
    source_lineage: dict[str, object],
    split_parameters: DatasetSplitParameters,
) -> tuple[Path, dict[str, object]]:
    if not samples:
        exclusion_counts = dict(
            sorted(
                Counter(
                    str(item.get("reason", "unspecified"))
                    for item in generation_exclusions
                ).items()
            )
        )
        examples = [
            {
                "asset": item.get("asset"),
                "decision_time": item.get("decision_time"),
                "reason": item.get("reason"),
            }
            for item in generation_exclusions[:3]
        ]
        raise ValueError(
            "No complete feature and label samples are available; "
            f"exclusion_counts={exclusion_counts}; examples={examples}"
        )
    resolutions = {sample.features.resolution for sample in samples}
    if len(resolutions) != 1:
        raise ValueError("A versioned dataset must use one candle resolution")
    parameters = samples[0].label.parameters
    if any(sample.label.parameters != parameters for sample in samples):
        raise ValueError("All labels in a dataset must use the same parameters")
    splits = split_chronologically(
        samples,
        resolution=next(iter(resolutions)),
        parameters=split_parameters,
    )
    ordered_samples = sorted(samples, key=_sample_key)
    generation_exclusions = sorted(
        generation_exclusions,
        key=lambda item: (
            str(item.get("decision_time", "")),
            str(item.get("asset", "")),
            str(item.get("reason", "")),
        ),
    )
    split_exclusions = [
        {
            "reason": excluded.reason,
            "sample": _sample_payload(excluded.sample),
        }
        for excluded in splits.excluded
    ]
    feature_schema_version = (
        samples[0].features.schema_version if samples else FEATURE_SCHEMA_VERSION
    )
    fingerprint_input = {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "feature_schema_version": feature_schema_version,
        "label_algorithm_version": TRIPLE_BARRIER_VERSION,
        "source_lineage": source_lineage,
        "label_parameters": parameters.model_dump(mode="json"),
        "split_parameters": split_parameters.model_dump(mode="json"),
        "split_boundaries": {
            "validation_start": splits.validation_start.isoformat(),
            "test_start": splits.test_start.isoformat(),
        },
        "samples": [_sample_payload(sample) for sample in ordered_samples],
        "generation_exclusions": generation_exclusions,
        "split_exclusions": split_exclusions,
    }
    dataset_digest = hashlib.sha256(
        _canonical_json(fingerprint_input).encode("utf-8")
    ).hexdigest()
    dataset_version = f"dataset-{dataset_digest[:24]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / dataset_version
    if target_dir.exists():
        manifest_path = target_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_version") == dataset_version:
            for filename, expected_hash in manifest.get("output_sha256", {}).items():
                output_path = target_dir / filename
                if (
                    not output_path.is_file()
                    or hashlib.sha256(output_path.read_bytes()).hexdigest()
                    != expected_hash
                ):
                    raise ValueError(
                        f"Existing dataset output failed checksum validation: "
                        f"{output_path}"
                    )
            return target_dir, manifest
        raise FileExistsError(f"Dataset output already exists: {target_dir}")

    split_rows = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    all_exclusions = generation_exclusions + split_exclusions
    exclusion_counts = dict(
        sorted(
            Counter(str(item.get("reason", "unspecified")) for item in all_exclusions)
            .items()
        )
    )
    manifest: dict[str, object] = {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "label_algorithm_version": TRIPLE_BARRIER_VERSION,
        "source_lineage": source_lineage,
        "label_parameters": parameters.model_dump(mode="json"),
        "split_parameters": split_parameters.model_dump(mode="json"),
        "split_boundaries": {
            "validation_start": splits.validation_start.isoformat(),
            "test_start": splits.test_start.isoformat(),
        },
        "sample_counts": {
            "input_labeled_complete": len(samples),
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
            "excluded": len(all_exclusions),
            "excluded_by_reason": exclusion_counts,
        },
        "built_at": datetime.now(UTC).isoformat(),
        "output_sha256": {},
    }
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{dataset_version}-", dir=output_dir)
    )
    try:
        output_hashes: dict[str, str] = {}
        for split_name, split_samples in split_rows.items():
            path = temporary_dir / f"{split_name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as output_file:
                for sample in split_samples:
                    output_file.write(_canonical_json(_sample_payload(sample)) + "\n")
            output_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

        excluded_path = temporary_dir / "excluded.jsonl"
        with excluded_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for item in generation_exclusions:
                output_file.write(_canonical_json(item) + "\n")
            for item in split_exclusions:
                output_file.write(_canonical_json(item) + "\n")
        output_hashes[excluded_path.name] = hashlib.sha256(
            excluded_path.read_bytes()
        ).hexdigest()
        manifest["output_sha256"] = output_hashes
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary_dir.rename(target_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return target_dir, manifest
