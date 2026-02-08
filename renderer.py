import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid
import json
from datetime import datetime, timedelta
from scipy.interpolate import PchipInterpolator
import os

def generate_chart(nested_data, output_path="chart.png"):
    # 1. TEMPORAL ANCHOR
    start_time = datetime(2026, 1, 28, 18, 0)

    # --- REFRESHED ENGINE LOGIC ---
    def flatten_data(nested):
        rows = []
        for block in nested:
            ts = pd.to_datetime(block["timestamp"])
            row = {"timestamp": ts}
            for entry in block["entries"]:
                row[entry["key"]] = entry["value"]
                if entry["key"] in ["glucose", "ketones", "body_weight"]:
                    row[f"is_{entry['key']}_simulated"] = entry.get("simulated", False)
            rows.append(row)
        return pd.DataFrame(rows).sort_values('timestamp')

    df = flatten_data(nested_data)
    # Ensure mandatory/expected columns exist
    for col in ['glucose', 'ketones', 'body_weight', 'cheat_snack', 'keto_snack']:
        if col not in df.columns:
            df[col] = np.nan
    for col in ['is_glucose_simulated', 'is_ketones_simulated', 'is_body_weight_simulated']:
        if col not in df.columns:
            df[col] = False

    # Handle timezone awareness for hours calculation
    df['hours_elapsed'] = df['timestamp'].apply(lambda x: (x.replace(tzinfo=None) - start_time).total_seconds() / 3600)

    # Calculate GKI safely (where both exist)
    df['gki'] = df.apply(lambda r: (r['glucose'] / 18.016) / r['ketones'] if pd.notnull(r['glucose']) and pd.notnull(r['ketones']) else np.nan, axis=1)

    # 3. INTERPOLATION ENGINE (PCHIP for chemistry)
    sim_hours = np.linspace(df['hours_elapsed'].min(), df['hours_elapsed'].max(), 1000)
    sim_dates = [start_time + timedelta(hours=h) for h in sim_hours]

    def get_pchip(sub_df, y_column, target_x):
        sub = sub_df.dropna(subset=[y_column])
        if len(sub) < 2: return np.zeros_like(target_x)
        return PchipInterpolator(sub['hours_elapsed'], sub[y_column])(target_x)

    smooth_glucose = get_pchip(df, 'glucose', sim_hours)
    smooth_ketones = get_pchip(df, 'ketones', sim_hours)
    smooth_gki = get_pchip(df, 'gki', sim_hours)

    # WEIGHT SIMULATION (Physiological Decay Model)
    # Connecting the measured points
    weight_df = df.dropna(subset=['body_weight'])
    if len(weight_df) >= 2:
        weight_start = weight_df.iloc[0]['body_weight']
        weight_final = weight_df.iloc[-1]['body_weight']
        k_decay = 0.008 
        smooth_weight = weight_final + (weight_start - weight_final) * np.exp(-k_decay * (sim_hours - weight_df.iloc[0]['hours_elapsed']))
    else:
        smooth_weight = np.zeros_like(sim_hours)

    # 4. CIRCADIAN GLUCOSE BANDING
    def get_circadian_band(dates):
        lows, highs = [], []
        for d in dates:
            hour = d.hour + d.minute/60.0
            # Model the Dawn Phenomenon (Peak at 7:00 AM)
            surge = 15 * np.exp(-((hour - 7)**2) / (2 * 1.5**2))
            lows.append(70 + surge * 0.5)
            highs.append(100 + surge)
        return np.array(lows), np.array(highs)

    band_low, band_high = get_circadian_band(sim_dates)

    # 5. REFEED & BRIDGE ANNOTATIONS
    refeed_events = df[df['cheat_snack'].notnull()]
    bridge_events = df[df['keto_snack'].notnull()]

    # 6. PLOT GENERATION
    plt.rcParams.update({'font.size': 28, 'font.family': 'sans-serif'})
    # Target 16,000px width (Max hardware texture size)
    # 50 inches * 320 DPI = 16,000 pixels
    fig, ax1 = plt.subplots(figsize=(50, 18.75), dpi=320) 
    plt.subplots_adjust(right=0.88, left=0.06, top=0.9, bottom=0.1)

    # Primary Axis: Glucose
    ax1.set_ylabel('Glucose (mg/dL) [Measured]', color='#d62728', fontweight='bold', fontsize=36)
    ax1.fill_between(sim_dates, band_low, band_high, color='red', alpha=0.08, label='Circadian Healthy Band')
    ax1.plot(sim_dates, smooth_glucose, color='#d62728', lw=10, alpha=0.2)
    ax1.scatter(df.dropna(subset=['glucose'])['timestamp'], df.dropna(subset=['glucose'])['glucose'], color='#d62728', s=400, edgecolors='black', label='Measured Glucose', zorder=5)
    ax1.set_ylim(40, 160)

    # Secondary Axis: Ketones
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ketones (mmol/L) [Measured]', color='#1f77b4', fontweight='bold')
    ax2.plot(sim_dates, smooth_ketones, color='#1f77b4', lw=10, alpha=0.2)
    ax2.scatter(df.dropna(subset=['ketones'])['timestamp'], df.dropna(subset=['ketones'])['ketones'], marker='s', color='#1f77b4', s=400, edgecolors='black', label='Measured Ketones', zorder=5)
    ax2.set_ylim(0, 10)

    # Tertiary Axis: GKI
    ax_gki = ax1.twinx()
    ax_gki.spines['right'].set_position(('outward', 60))
    ax_gki.set_ylabel('GKI (Repair Index) [Computed]', color='#9467bd', fontweight='bold')
    ax_gki.plot(sim_dates, smooth_gki, color='#9467bd', lw=10, alpha=0.2, ls='--')
    ax_gki.scatter(df.dropna(subset=['gki'])['timestamp'], df.dropna(subset=['gki'])['gki'], marker='D', color='#9467bd', s=300, edgecolors='black', label='Computed GKI', zorder=5)
    ax_gki.axhline(y=1.0, color='purple', ls='-', lw=5, alpha=0.7, label='Mitophagy Goal (1.0)')
    ax_gki.fill_between(sim_dates, 0, 1.0, color='purple', alpha=0.1)
    ax_gki.set_ylim(0, 10)

    # Quaternary Axis: Weight
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))
    ax3.set_ylabel('Body Weight (lbs) [Simulated Path]', color='#2ca02c', fontweight='bold')
    ax3.plot(sim_dates, smooth_weight, color='#2ca02c', lw=12, alpha=0.5, label='Simulation Path')

    # Markers for ground truth weight
    weight_meas = df[(df['body_weight'].notnull()) & (df.get('is_body_weight_simulated', pd.Series([False]*len(df))) == False)]
    ax3.scatter(weight_meas['timestamp'], weight_meas['body_weight'], marker='^', color='#2ca02c', s=600, edgecolors='black', label='Measured Anchor', zorder=6)
    ax3.axhline(y=220, color='blue', ls='-.', lw=4, alpha=0.5, label='Obesity Exit: 220')
    ax3.set_ylim(170, 240)

    # Annotate Refeeds and Bridges
    for idx, row in refeed_events.iterrows():
        ax1.axvline(x=row['timestamp'], color='salmon', lw=4, alpha=0.6, ls=':')
        ax1.text(row['timestamp'], 158, row['cheat_snack'], rotation=90, verticalalignment='top', fontsize=10, fontweight='bold')

    for _, row in bridge_events.iterrows():
        ax1.axvline(x=row['timestamp'], color='lightgreen', lw=4, alpha=0.6, ls=':')
        ax1.text(row['timestamp'], 158, row['keto_snack'], rotation=90, verticalalignment='top', fontsize=12, fontweight='bold')

    plt.title("Master-View v28: FastTrack Dashboard Analytics", fontsize=60, fontweight='bold', pad=120)
    plt.savefig(output_path, dpi=320) 
    plt.close()
