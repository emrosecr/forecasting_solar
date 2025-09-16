"""
Visualization helpers for SHAP values and distance-of-impact summary.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_shap_summary(importance_df: pd.DataFrame, top_n: int = 30, output_path: Optional[str] = None):
    df = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(6, 0.3 * len(df))))
    plt.barh(df['feature'], df['mean_abs_shap'], color='#1f77b4')
    plt.xlabel('Mean |SHAP|')
    plt.ylabel('Feature')
    plt.title('SHAP Feature Importance (Top)')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_distance_of_impact(ring_summary: pd.DataFrame, ring_width_km: float, output_path: Optional[str] = None):
    df = ring_summary.copy()
    # Convert ring index to mid-distance (km)
    df['distance_km'] = (df['ring'] - 0.5) * ring_width_km
    plt.figure(figsize=(8, 5))
    plt.plot(df['distance_km'], df['total_importance'], marker='o')
    plt.xlabel('Distance from Site (km)')
    plt.ylabel('Total Mean |SHAP| per Ring')
    plt.title('Distance-of-Impact of Climate Drivers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

