"""
Singapore Smart City — Model Drift Monitor

Statistical drift detection using:
- PSI (Population Stability Index) for data drift
- KS Test for prediction/confidence drift
- Correlation tracking for concept drift

No ML models — pure statistics for trustworthiness and explainability.
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# =============================================================================
# Drift Detection Methods
# =============================================================================

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: No significant change
    0.1 ≤ PSI < 0.2: Moderate change — investigate
    PSI ≥ 0.2: Significant change — action required

    Args:
        baseline: Reference distribution (array of values).
        current: Current distribution to compare.
        n_bins: Number of bins for histogram comparison.

    Returns:
        PSI value (float).
    """
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Create bins from baseline
    min_val = min(baseline.min(), current.min())
    max_val = max(baseline.max(), current.max())

    if min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Compute proportions
    baseline_hist, _ = np.histogram(baseline, bins=bins)
    current_hist, _ = np.histogram(current, bins=bins)

    # Normalize to proportions (avoid division by zero)
    baseline_prop = (baseline_hist + 1e-6) / (len(baseline) + n_bins * 1e-6)
    current_prop = (current_hist + 1e-6) / (len(current) + n_bins * 1e-6)

    # PSI formula
    psi = np.sum(
        (current_prop - baseline_prop) * np.log(current_prop / baseline_prop)
    )

    return round(float(psi), 6)


def compute_ks_test(
    baseline: np.ndarray,
    current: np.ndarray,
) -> dict:
    """Run Kolmogorov-Smirnov test between two distributions.

    Args:
        baseline: Reference distribution.
        current: Current distribution.

    Returns:
        Dict with statistic, p_value, and drift flag.
    """
    if len(baseline) < 5 or len(current) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    stat, p_value = scipy_stats.ks_2samp(baseline, current)

    return {
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "drift_detected": p_value < 0.05,
    }


# =============================================================================
# Drift Monitor
# =============================================================================

@dataclass
class DriftAlert:
    """A single drift detection alert."""
    timestamp: str
    drift_type: str           # "data", "prediction", "concept"
    metric_name: str          # What was measured
    severity: str             # "low", "moderate", "high"
    value: float              # The metric value
    threshold: float          # The threshold that was exceeded
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DriftMonitor:
    """Monitors model and data drift over time using statistical methods."""

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Drift detection configuration. Uses defaults if None.
        """
        drift_config = (config or {}).get("drift_detection", {})

        self.psi_threshold = drift_config.get("psi_threshold", 0.2)
        self.ks_alpha = drift_config.get("ks_test_alpha", 0.05)
        self.window_minutes = drift_config.get("rolling_window_minutes", 60)
        self.correlation_sigma = drift_config.get("correlation_sigma", 2.0)

        # Baseline distributions (set from initial data)
        self.baseline_brightness = None
        self.baseline_confidence = None
        self.baseline_vehicle_count = None
        self.baseline_correlation = None  # brightness vs confidence

        # Rolling windows for current data
        self.current_brightness = deque(maxlen=5000)
        self.current_confidence = deque(maxlen=5000)
        self.current_vehicle_count = deque(maxlen=5000)

        # Alert history
        self.alerts: list[DriftAlert] = []

    def set_baseline(self, detection_results: list[dict]):
        """Set baseline distributions from initial stable data.

        Call this once with your first batch of detection results
        (collected under "normal" conditions).

        Args:
            detection_results: List of detection result dicts.
        """
        brightness = []
        confidence = []
        vehicle_counts = []

        for r in detection_results:
            brightness.append(r.get("mean_brightness", 128))
            if r.get("mean_confidence", 0) > 0:
                confidence.append(r["mean_confidence"])
            vehicle_counts.append(r.get("num_vehicles", 0))

        self.baseline_brightness = np.array(brightness)
        self.baseline_confidence = np.array(confidence)
        self.baseline_vehicle_count = np.array(vehicle_counts)

        # Compute baseline correlation (brightness vs confidence)
        if len(brightness) > 10 and len(confidence) > 10:
            min_len = min(len(brightness), len(confidence))
            corr, _ = scipy_stats.pearsonr(
                brightness[:min_len], confidence[:min_len]
            )
            self.baseline_correlation = corr
        else:
            self.baseline_correlation = 0.0

        logger.info(
            f"Baseline set: {len(detection_results)} frames, "
            f"mean brightness={np.mean(brightness):.1f}, "
            f"mean confidence={np.mean(confidence):.3f}"
        )

    def check_drift(
        self,
        detection_results: list[dict],
        timestamp: str = "",
    ) -> list[DriftAlert]:
        """Check for drift in a batch of new detection results.

        Args:
            detection_results: New batch of detection results.
            timestamp: ISO timestamp for this check.

        Returns:
            List of DriftAlert objects (empty if no drift detected).
        """
        if self.baseline_brightness is None:
            logger.warning("No baseline set — call set_baseline() first")
            return []

        alerts = []
        ts = timestamp or datetime.now().isoformat()

        # Extract current distributions
        brightness = np.array([r.get("mean_brightness", 128) for r in detection_results])
        confidence = np.array([
            r["mean_confidence"] for r in detection_results
            if r.get("mean_confidence", 0) > 0
        ])
        vehicle_counts = np.array([r.get("num_vehicles", 0) for r in detection_results])

        # Update rolling windows
        self.current_brightness.extend(brightness.tolist())
        self.current_confidence.extend(confidence.tolist())
        self.current_vehicle_count.extend(vehicle_counts.tolist())

        # --- 1. Data Drift (PSI on brightness) ---
        psi_brightness = compute_psi(self.baseline_brightness, brightness)
        if psi_brightness >= self.psi_threshold:
            severity = "high" if psi_brightness >= 0.3 else "moderate"
            alert = DriftAlert(
                timestamp=ts,
                drift_type="data",
                metric_name="image_brightness_psi",
                severity=severity,
                value=psi_brightness,
                threshold=self.psi_threshold,
                details={
                    "baseline_mean": round(float(self.baseline_brightness.mean()), 1),
                    "current_mean": round(float(brightness.mean()), 1),
                    "cause": self._diagnose_brightness_drift(brightness),
                },
            )
            alerts.append(alert)

        # --- 2. Prediction Drift (KS test on confidence) ---
        if len(confidence) >= 5:
            ks_result = compute_ks_test(self.baseline_confidence, confidence)
            if ks_result["drift_detected"]:
                alert = DriftAlert(
                    timestamp=ts,
                    drift_type="prediction",
                    metric_name="confidence_ks_test",
                    severity="moderate" if ks_result["p_value"] > 0.01 else "high",
                    value=ks_result["statistic"],
                    threshold=self.ks_alpha,
                    details={
                        "p_value": ks_result["p_value"],
                        "baseline_mean_conf": round(float(self.baseline_confidence.mean()), 3),
                        "current_mean_conf": round(float(confidence.mean()), 3),
                        "cause": self._diagnose_confidence_drift(confidence),
                    },
                )
                alerts.append(alert)

        # --- 3. Concept Drift (correlation shift) ---
        if len(brightness) > 10 and len(confidence) > 10:
            min_len = min(len(brightness), len(confidence))
            current_corr, _ = scipy_stats.pearsonr(
                brightness[:min_len], confidence[:min_len]
            )

            if self.baseline_correlation is not None:
                corr_shift = abs(current_corr - self.baseline_correlation)
                # Use historical data to estimate what's "normal" shift
                if corr_shift > 0.3:  # Significant correlation change
                    alert = DriftAlert(
                        timestamp=ts,
                        drift_type="concept",
                        metric_name="brightness_confidence_correlation",
                        severity="high" if corr_shift > 0.5 else "moderate",
                        value=round(current_corr, 3),
                        threshold=round(self.baseline_correlation, 3),
                        details={
                            "baseline_correlation": round(float(self.baseline_correlation), 3),
                            "current_correlation": round(current_corr, 3),
                            "shift": round(corr_shift, 3),
                            "cause": "Relationship between image quality and detection confidence has changed",
                        },
                    )
                    alerts.append(alert)

        # Store alerts
        self.alerts.extend(alerts)

        if alerts:
            for a in alerts:
                logger.warning(
                    f"🚨 DRIFT ALERT: {a.drift_type}/{a.metric_name} "
                    f"= {a.value} (threshold: {a.threshold}) — {a.severity}"
                )
        else:
            logger.debug("No drift detected in this batch")

        return alerts

    def _diagnose_brightness_drift(self, current: np.ndarray) -> str:
        """Suggest a root cause for brightness drift."""
        baseline_mean = float(self.baseline_brightness.mean())
        current_mean = float(current.mean())

        if current_mean < baseline_mean * 0.7:
            return "Likely cause: time-of-day shift (night) or camera degradation"
        elif current_mean > baseline_mean * 1.3:
            return "Likely cause: time-of-day shift (bright daylight) or camera overexposure"
        else:
            return "Likely cause: weather conditions or gradual camera sensor drift"

    def _diagnose_confidence_drift(self, current: np.ndarray) -> str:
        """Suggest a root cause for confidence drift."""
        baseline_mean = float(self.baseline_confidence.mean())
        current_mean = float(current.mean())

        if current_mean < baseline_mean * 0.8:
            return "Model confidence dropped — possible domain shift, weather, or camera issue"
        elif current_mean > baseline_mean * 1.2:
            return "Model confidence increased — possible data simplification or fewer objects"
        else:
            return "Confidence distribution shape changed — investigate specific cameras"

    def get_drift_summary(self) -> dict:
        """Get summary of all drift alerts."""
        return {
            "total_alerts": len(self.alerts),
            "by_type": {
                "data": len([a for a in self.alerts if a.drift_type == "data"]),
                "prediction": len([a for a in self.alerts if a.drift_type == "prediction"]),
                "concept": len([a for a in self.alerts if a.drift_type == "concept"]),
            },
            "by_severity": {
                "high": len([a for a in self.alerts if a.severity == "high"]),
                "moderate": len([a for a in self.alerts if a.severity == "moderate"]),
                "low": len([a for a in self.alerts if a.severity == "low"]),
            },
            "latest_alerts": [a.to_dict() for a in self.alerts[-5:]],
        }
