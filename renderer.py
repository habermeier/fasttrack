import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
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
    generated_at_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

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
    for col in ['glucose', 'ketones', 'body_weight', 'total_fat', 'visceral_fat', 'water_percent', 'cheat_snack', 'keto_snack']:
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

    def mask_before_first_point(measured_df, target_x, values):
        if len(measured_df) == 0:
            return np.full_like(values, np.nan, dtype=float)
        first_hour = float(measured_df['hours_elapsed'].iloc[0])
        return np.where(target_x >= first_hour, values, np.nan)

    is_glucose_measured = (df.get('is_glucose_simulated', pd.Series(False, index=df.index)) == False)
    is_ketones_measured = (df.get('is_ketones_simulated', pd.Series(False, index=df.index)) == False)
    is_weight_measured = (df.get('is_body_weight_simulated', pd.Series(False, index=df.index)) == False)

    glucose_meas_df = df[df['glucose'].notnull() & is_glucose_measured].sort_values('hours_elapsed')
    ketones_meas_df = df[df['ketones'].notnull() & is_ketones_measured].sort_values('hours_elapsed')
    gki_meas_df = df[df['gki'].notnull() & is_glucose_measured & is_ketones_measured].sort_values('hours_elapsed')
    weight_meas_df = df[df['body_weight'].notnull() & is_weight_measured].sort_values('hours_elapsed')

    smooth_glucose = get_pchip(df, 'glucose', sim_hours)
    smooth_ketones = get_pchip(df, 'ketones', sim_hours)
    smooth_gki = get_pchip(df, 'gki', sim_hours)

    smooth_glucose_masked = mask_before_first_point(glucose_meas_df, sim_hours, smooth_glucose)
    smooth_ketones_masked = mask_before_first_point(ketones_meas_df, sim_hours, smooth_ketones)
    smooth_gki_masked = mask_before_first_point(gki_meas_df, sim_hours, smooth_gki)

    def get_anchor_predictive_series(sub_df, y_column, target_x, tail_damping=0.35):
        sub = sub_df.dropna(subset=[y_column]).sort_values('hours_elapsed')
        if len(sub) == 0:
            return np.zeros_like(target_x), sub
        if len(sub) == 1:
            return np.full_like(target_x, float(sub.iloc[0][y_column])), sub

        x = sub['hours_elapsed'].astype(float).to_numpy()
        y = sub[y_column].astype(float).to_numpy()

        # Anchor-preserving interpolation: curve passes through all measured points.
        pchip = PchipInterpolator(x, y)
        smooth = np.empty_like(target_x, dtype=float)

        inside = (target_x >= x[0]) & (target_x <= x[-1])
        smooth[inside] = pchip(target_x[inside])

        left = target_x < x[0]
        if np.any(left):
            dx = x[1] - x[0]
            slope_left = (y[1] - y[0]) / dx if dx != 0 else 0.0
            smooth[left] = y[0] + slope_left * tail_damping * (target_x[left] - x[0])

        right = target_x > x[-1]
        if np.any(right):
            dx = x[-1] - x[-2]
            slope_right = (y[-1] - y[-2]) / dx if dx != 0 else 0.0
            smooth[right] = y[-1] + slope_right * tail_damping * (target_x[right] - x[-1])

        # Keep tails bounded near observed range.
        band = max(2.0, 0.1 * (y.max() - y.min()))
        smooth = np.clip(smooth, y.min() - band, y.max() + band)
        return smooth, sub

    # Weight hybrid model: pin to measured anchors, predict only outside anchor range.
    smooth_weight, _ = get_anchor_predictive_series(df, 'body_weight', sim_hours)
    smooth_weight_masked = mask_before_first_point(weight_meas_df, sim_hours, smooth_weight)

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
    water_df = df.dropna(subset=['water_percent'])
    total_fat_df = df.dropna(subset=['total_fat']).sort_values('hours_elapsed')
    visceral_fat_df = df.dropna(subset=['visceral_fat']).sort_values('hours_elapsed')
    has_water = len(water_df) > 0
    has_body_fat = len(total_fat_df) > 0 or len(visceral_fat_df) > 0

    # PLOT GENERATION
    plt.rcParams.update({'font.size': 28, 'font.family': 'sans-serif'})
    
    # Dynamic Width Calculation: ~10 inches per day, minimum 50 inches.
    total_days = max(1, (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400)
    default_dynamic_width = max(50.0, total_days * 10.0)
    
    chart_width = float(os.getenv('FASTTRACK_CHART_WIDTH_IN', str(default_dynamic_width)))
    chart_height = float(os.getenv('FASTTRACK_CHART_HEIGHT_IN', '18.75'))
    chart_dpi = int(os.getenv('FASTTRACK_CHART_DPI', '320'))
    fig, ax1 = plt.subplots(figsize=(chart_width, chart_height), dpi=chart_dpi)
    if has_water and has_body_fat:
        plot_right = 0.58
    elif has_water or has_body_fat:
        plot_right = 0.68
    else:
        plot_right = 0.80
    plt.subplots_adjust(right=plot_right, left=0.06, top=0.96, bottom=0.1)

    # Primary Axis: Glucose
    ax1.set_ylabel('Glucose (mg/dL) [Measured]', color='#d62728', fontweight='bold', fontsize=36)
    ax1.fill_between(
        sim_dates,
        band_low,
        band_high,
        color='#ff7f50',
        alpha=0.18,
        zorder=0,
        label='Expected Glucose Band'
    )
    ax1.plot(sim_dates, band_low, color='#ff7f50', lw=2.5, alpha=0.45, ls=':')
    ax1.plot(sim_dates, band_high, color='#ff7f50', lw=2.5, alpha=0.45, ls=':')
    ax1.plot(sim_dates, smooth_glucose_masked, color='#d62728', lw=10, alpha=0.2)
    ax1.scatter(glucose_meas_df['timestamp'], glucose_meas_df['glucose'], color='#d62728', s=400, edgecolors='black', label='Measured Glucose', zorder=5)
    ax1.set_ylim(40, 160)
    ax1.tick_params(axis='y', colors='#d62728', labelsize=24)

    # Secondary Axis: Ketones
    ax2 = ax1.twinx()
    ax2.set_ylabel('Ketones / GKI (0-10)', color='#1f77b4', fontweight='bold', fontsize=24, labelpad=18)
    ax2.plot(sim_dates, smooth_ketones_masked, color='#1f77b4', lw=10, alpha=0.2)
    ax2.scatter(ketones_meas_df['timestamp'], ketones_meas_df['ketones'], marker='s', color='#1f77b4', s=400, edgecolors='black', label='Measured Ketones', zorder=5)
    ax2.plot(sim_dates, smooth_gki_masked, color='#9467bd', lw=10, alpha=0.2, ls='--')
    ax2.scatter(gki_meas_df['timestamp'], gki_meas_df['gki'], marker='D', color='#9467bd', s=300, edgecolors='black', label='Computed GKI', zorder=5)
    ax2.axhline(y=1.0, color='purple', ls='-', lw=5, alpha=0.7, label='Mitophagy Goal (1.0)')
    ax2.fill_between(sim_dates, 0, 1.0, color='purple', alpha=0.1)
    ax2.set_ylim(0, 10)
    ax2.yaxis.set_major_locator(MultipleLocator(2))
    ax2.tick_params(axis='y', colors='#1f77b4', labelsize=22, pad=4)

    # Quaternary Axis: Weight
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 90))
    ax3.set_ylabel('Weight (lbs)', color='#2ca02c', fontweight='bold', fontsize=24, labelpad=14)
    ax3.plot(sim_dates, smooth_weight_masked, color='#2ca02c', lw=12, alpha=0.5, label='Simulation Path')

    # Markers for ground truth weight
    ax3.scatter(weight_meas_df['timestamp'], weight_meas_df['body_weight'], marker='^', color='#2ca02c', s=600, edgecolors='black', label='Measured Anchor', zorder=6)
    ax3.axhline(y=220, color='blue', ls='-.', lw=4, alpha=0.5, label='Obesity Exit: 220')
    ax3.set_ylim(170, 240)
    ax3.yaxis.set_major_locator(MultipleLocator(10))
    ax3.tick_params(axis='y', colors='#2ca02c', labelsize=22, pad=3)

    # Optional Axis: Body Fat %
    if has_body_fat:
        ax_fat = ax1.twinx()
        ax_fat.spines['right'].set_position(('outward', 170 if has_water else 90))
        ax_fat.set_ylabel('Body/Visceral Fat (%)', color='#ff8c00', fontweight='bold', fontsize=24, labelpad=12)

        # Start lines at first recorded data point (no back-fill before first measurement).
        if len(total_fat_df) >= 2:
            smooth_total_fat = get_pchip(df, 'total_fat', sim_hours)
            total_masked = mask_before_first_point(total_fat_df, sim_hours, smooth_total_fat)
            ax_fat.plot(sim_dates, total_masked, color='#ff8c00', lw=8, alpha=0.4, ls='-')
        if len(visceral_fat_df) >= 2:
            smooth_visceral_fat = get_pchip(df, 'visceral_fat', sim_hours)
            visceral_masked = mask_before_first_point(visceral_fat_df, sim_hours, smooth_visceral_fat)
            ax_fat.plot(sim_dates, visceral_masked, color='#8b4513', lw=8, alpha=0.4, ls='--')

        ax_fat.scatter(
            total_fat_df['timestamp'],
            total_fat_df['total_fat'],
            marker='P',
            color='#ff8c00',
            s=320,
            edgecolors='black',
            zorder=6
        )
        ax_fat.scatter(
            visceral_fat_df['timestamp'],
            visceral_fat_df['visceral_fat'],
            marker='X',
            color='#8b4513',
            s=320,
            edgecolors='black',
            zorder=6
        )

        fat_values = []
        if len(total_fat_df) > 0:
            fat_values.extend(total_fat_df['total_fat'].astype(float).tolist())
        if len(visceral_fat_df) > 0:
            fat_values.extend(visceral_fat_df['visceral_fat'].astype(float).tolist())
        fat_min = min(fat_values)
        fat_max = max(fat_values)
        if fat_min == fat_max:
            fat_min -= 2
            fat_max += 2
        ax_fat.set_ylim(max(0, fat_min - 2), fat_max + 2)
        ax_fat.yaxis.set_major_locator(MultipleLocator(2))
        ax_fat.tick_params(axis='y', colors='#ff8c00', labelsize=22, pad=2)

    # Optional Axis: Water %
    if has_water:
        ax_water = ax1.twinx()
        ax_water.spines['right'].set_position(('outward', 250 if has_body_fat else 170))
        ax_water.set_ylabel('Water (%)', color='#17becf', fontweight='bold', fontsize=24, labelpad=12)
        if len(water_df) >= 2:
            smooth_water = get_pchip(df, 'water_percent', sim_hours)
            smooth_water_masked = mask_before_first_point(water_df, sim_hours, smooth_water)
            ax_water.plot(sim_dates, smooth_water_masked, color='#17becf', lw=8, alpha=0.35, ls='-.')
        ax_water.scatter(
            water_df['timestamp'],
            water_df['water_percent'],
            marker='o',
            color='#17becf',
            s=280,
            edgecolors='black',
            zorder=6
        )
        water_min = float(water_df['water_percent'].min())
        water_max = float(water_df['water_percent'].max())
        if water_min == water_max:
            water_min -= 3
            water_max += 3
        ax_water.set_ylim(max(40, water_min - 2), min(80, water_max + 2))
        ax_water.yaxis.set_major_locator(MultipleLocator(2))
        ax_water.tick_params(axis='y', colors='#17becf', labelsize=22, pad=2)

    # Annotate Refeeds and Bridges with 1-hour wide markers (no text labels)
    for _, row in refeed_events.iterrows():
        # 1-hour wide marker (30 min on each side)
        span_start = row['timestamp'] - timedelta(minutes=30)
        span_end = row['timestamp'] + timedelta(minutes=30)
        ax1.axvspan(span_start, span_end, color='salmon', alpha=0.24, zorder=1)

    for _, row in bridge_events.iterrows():
        # 1-hour wide marker (30 min on each side)
        span_start = row['timestamp'] - timedelta(minutes=30)
        span_end = row['timestamp'] + timedelta(minutes=30)
        ax1.axvspan(span_start, span_end, color='lightgreen', alpha=0.24, zorder=1)

    # Keep date labels legible on dense windows.
    # Aim for approximately one tick every 5 inches.
    max_ticks = max(9, int(chart_width / 5))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=max_ticks))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', labelsize=22)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

    fig.text(
        0.995,
        0.01,
        f"Generated: {generated_at_utc}",
        ha='right',
        va='bottom',
        fontsize=20,
        color='#333333',
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.25')
    )
    plt.savefig(output_path, dpi=chart_dpi)
    plt.close()
