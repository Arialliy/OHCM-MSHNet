from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "mIoU": ("mIoU", "miou", "mean_iou", "mean_IoU", "MeanIoU", "iou"),
    "FA_ppm": (
        "FA_ppm",
        "fa_ppm",
        "FAppm",
        "FA ppm",
        "FA",
        "false_alarm_ppm",
        "false_alarm_rate_ppm",
    ),
    "Precision": ("Precision", "precision", "Prec", "prec"),
    "Pd": ("Pd", "pd", "PD", "target_pd", "detection_probability"),
}

NESTED_METRIC_KEYS = (
    "metrics",
    "metric",
    "metrics_at_threshold",
    "summary_metrics",
    "aggregate",
    "overall",
    "result",
    "results",
    "fixed_threshold",
    "at_threshold",
    "threshold_0.5",
)


@dataclass(frozen=True)
class MetricRecord:
    mIoU: float
    FA_ppm: float
    Precision: float
    Pd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DeltaRecord:
    mIoU: float
    FA_ppm: float
    Precision: float
    Pd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _norm_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _candidate_metric_maps(summary: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield summary
    for key in NESTED_METRIC_KEYS:
        value = summary.get(key)
        if isinstance(value, Mapping):
            yield value


def get_metric(summary: Mapping[str, Any], canonical_name: str) -> float:
    if canonical_name not in METRIC_ALIASES:
        raise KeyError(f"Unknown metric canonical name: {canonical_name}")

    aliases = METRIC_ALIASES[canonical_name]
    normalized_aliases = {_norm_key(alias) for alias in aliases}

    visible_keys: list[str] = []
    for metric_map in _candidate_metric_maps(summary):
        items = list(metric_map.items())
        visible_keys.extend(str(key) for key, _ in items)
        for alias in aliases:
            normalized_alias = _norm_key(alias)
            for key, value in items:
                if key == alias or _norm_key(str(key)) == normalized_alias:
                    try:
                        return float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Metric {canonical_name} found as {key!r}, but value is not numeric: {value!r}"
                        ) from exc

    raise KeyError(f"Metric {canonical_name!r} not found. Available keys: {sorted(set(visible_keys))}")


def metrics_from_summary(summary: Mapping[str, Any]) -> MetricRecord:
    return MetricRecord(
        mIoU=get_metric(summary, "mIoU"),
        FA_ppm=get_metric(summary, "FA_ppm"),
        Precision=get_metric(summary, "Precision"),
        Pd=get_metric(summary, "Pd"),
    )


def load_metrics(path: str | Path) -> MetricRecord:
    return metrics_from_summary(load_json(path))


def delta_metrics(candidate: MetricRecord, baseline: MetricRecord) -> DeltaRecord:
    return DeltaRecord(
        mIoU=candidate.mIoU - baseline.mIoU,
        FA_ppm=candidate.FA_ppm - baseline.FA_ppm,
        Precision=candidate.Precision - baseline.Precision,
        Pd=candidate.Pd - baseline.Pd,
    )


def pass_hcval_improvement(
    delta: DeltaRecord,
    *,
    min_delta_miou: float = 0.005,
    min_fa_reduction: float = 10.0,
    min_delta_precision: float = 0.0,
    min_delta_pd: float = 0.0,
) -> bool:
    return (
        delta.mIoU >= min_delta_miou
        and delta.FA_ppm <= -min_fa_reduction
        and delta.Precision >= min_delta_precision
        and delta.Pd >= min_delta_pd
    )


def pass_nonregression(delta: DeltaRecord, *, eps: float = 1e-12) -> bool:
    return (
        delta.mIoU >= -eps
        and delta.FA_ppm <= eps
        and delta.Precision >= -eps
        and delta.Pd >= -eps
    )


def safe_positive_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"Empty name in named path: {value!r}")
    if not path:
        raise ValueError(f"Empty path in named path: {value!r}")
    return name, Path(path)


def load_named_metrics(values: list[str]) -> dict[str, MetricRecord]:
    records: dict[str, MetricRecord] = {}
    for value in values:
        name, path = parse_named_path(value)
        if name in records:
            raise ValueError(f"Duplicate named summary: {name}")
        records[name] = load_metrics(path)
    return records
