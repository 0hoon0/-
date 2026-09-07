"""Audit evidence & financial anomaly detection toolkit.

Implements two classic risk-assessment procedures from audit data analytics:
  * Benford's Law digit analysis (ISA 240/315 fraud risk screening)
  * 3-sigma statistical outlier screening on financial ratio trends
"""

from audit_anomaly.benford import BenfordResult, analyze_benford
from audit_anomaly.outliers import OutlierResult, detect_outliers_3sigma

__all__ = [
    "BenfordResult",
    "analyze_benford",
    "OutlierResult",
    "detect_outliers_3sigma",
]
