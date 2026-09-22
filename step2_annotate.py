"""
Step 2: Detect significant grade shifts and add feedback annotations.
- One annotated chart per subject per student
- Organized by class code (current class based on the latest date entry)
- Students tracked across ALL CSVs (handles class transfers)
- Only generates charts for students present in the latest date entry of a class
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = BASE_DIR / "sample_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "charts_annotated"
DEFAULT_PATTERN = "*.csv"

SUBJECTS = ["Vocabulary", "Science HW", "Science Test", "Reading", "Grammar Test", "Novel"]

COLORS = {
    "Vocabulary": "#e74c3c",
    "Science HW": "#f39c12",
    "Science Test": "#9b59b6",
    "Reading": "#e67e22",
    "Grammar Test": "#2980b9",
    "Novel": "#27ae60",
}

# Threshold for "significant change" (points difference week-over-week)
CHANGE_THRESHOLD = 15

# Fonts that can render Hangul, tried in order; the first one installed is used as a fallback.
CJK_FONT_CANDIDATES = [
    "Apple SD Gothic Neo", "AppleGothic",
    "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic",
]


def configure_fonts():
    """Add a Hangul-capable font as a fallback so the Korean half of each name renders."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = ["DejaVu Sans", name]
            return name
    return None


def load_all_data(csv_dir, pattern):
    """Load all CSV files and combine into one DataFrame."""
    csv_files = sorted(csv_dir.glob(pattern))
    if not csv_files:
        raise SystemExit(f"No CSV files matching '{pattern}' in {csv_dir}")
    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["Class"] = csv_file.stem
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], format="mixed")
    for subj in SUBJECTS:
        if subj in combined.columns:
            combined[subj] = pd.to_numeric(combined[subj], errors="coerce")
    return combined


def get_active_students(df):
    """
    Find students present in the latest date entry of each class.
    Returns dict: {student_name: current_class_code}
    """
    active = {}
    for cls in sorted(df["Class"].unique()):
        cls_data = df[df["Class"] == cls]
        latest_date = cls_data["Date"].max()
        latest_rows = cls_data[cls_data["Date"] == latest_date]
        for name in latest_rows["Name"].unique():
            if name not in active or latest_date > df[(df["Name"] == name) & (df["Class"] == active[name])]["Date"].max():
                active[name] = cls
    return active


def get_student_all_data(df, student_name):
    """Get ALL data for a student across all CSVs, sorted by date."""
    student_data = df[df["Name"] == student_name].copy()
    student_data = student_data.sort_values("Date")
    student_data = student_data.drop_duplicates(subset=["Date"], keep="last")
    return student_data


def get_best_fit(dates_numeric, values, num_points=100):
    """Polynomial least-squares fit (degree <= 3) used purely for visual smoothing."""
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return None, None
    x = dates_numeric[mask]
    y = values[mask]
    degree = min(3, len(x) - 1)
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_smooth = np.linspace(x.min(), x.max(), num_points)
        y_smooth = poly(x_smooth)
        y_smooth = np.clip(y_smooth, 0, 100)
        return x_smooth, y_smooth
    except Exception:
        return None, None


def detect_significant_changes(dates, values, threshold=CHANGE_THRESHOLD):
    """Detect points where the score moved by at least `threshold` since the previous recorded value."""
    changes = []
    valid_idx = ~np.isnan(values)
    valid_dates = dates[valid_idx].values
    valid_vals = values[valid_idx]

    if len(valid_vals) < 2:
        return changes

    for i in range(1, len(valid_vals)):
        diff = valid_vals[i] - valid_vals[i - 1]
        if abs(diff) >= threshold:
            direction = "up" if diff > 0 else "down"
            changes.append({
                "date": pd.Timestamp(valid_dates[i]),
                "value": valid_vals[i],
                "prev_value": valid_vals[i - 1],
                "change": diff,
                "direction": direction,
            })
    return changes


def generate_feedback(change):
    """Generate a feedback comment for a significant change."""
    diff = abs(change["change"])
    direction = change["direction"]
    prev = change["prev_value"]
    curr = change["value"]

    if direction == "up":
        if diff >= 30:
            comment = f"+{diff:.0f}pts! Major improvement"
        elif diff >= 20:
            comment = f"+{diff:.0f}pts, strong progress"
        else:
            comment = f"+{diff:.0f}pts, good gain"
    else:
        if diff >= 30:
            comment = f"-{diff:.0f}pts, significant drop"
        elif diff >= 20:
            comment = f"-{diff:.0f}pts, needs attention"
        else:
            comment = f"-{diff:.0f}pts, slight dip"

    return f"{comment} ({prev:.0f} -> {curr:.0f})"


def safe_filename(student_name):
    return student_name.replace(" ", "_").replace("*", "").replace("/", "-")


def create_annotated_subject_chart(student_name, student_data, subject, class_name, output_dir):
    """Create an annotated chart for one subject for one student."""
    vals = student_data[subject].values
    dates = student_data["Date"]
    dates_numeric = (dates - dates.min()).dt.total_seconds() / 86400

    valid = ~np.isnan(vals)
    if valid.sum() < 2:
        return None, 0

    color = COLORS.get(subject, "#333333")

    # Detect significant changes
    changes = detect_significant_changes(dates, vals)

    # Create figure with space for notes if needed
    if changes:
        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor("#fafafa")
        ax_chart = fig.add_axes([0.07, 0.25, 0.88, 0.65])
        ax_notes = fig.add_axes([0.07, 0.02, 0.88, 0.20])
    else:
        fig = plt.figure(figsize=(14, 7))
        fig.patch.set_facecolor("#fafafa")
        ax_chart = fig.add_axes([0.07, 0.10, 0.88, 0.80])
        ax_notes = None

    ax_chart.set_facecolor("#fafafa")

    # Raw data points
    ax_chart.scatter(dates[valid], vals[valid], color=color, alpha=0.4, s=40, zorder=2, label="Scores")

    # Best-fit line
    x_fit, y_fit = get_best_fit(dates_numeric.values, vals)
    if x_fit is not None:
        fit_dates = dates.min() + pd.to_timedelta(x_fit, unit="D")
        ax_chart.plot(fit_dates, y_fit, color=color, linewidth=2.5, label="Trend", zorder=3)

    # Add annotation markers
    for i, change in enumerate(changes):
        marker = "^" if change["direction"] == "up" else "v"
        marker_color = "#27ae60" if change["direction"] == "up" else "#e74c3c"

        ax_chart.scatter(change["date"], change["value"], color=marker_color,
                         marker=marker, s=120, zorder=5, edgecolors="white", linewidths=1.5)

        label_y = change["value"] + 4 if change["direction"] == "up" else change["value"] - 6
        ax_chart.annotate(
            str(i + 1),
            xy=(change["date"], change["value"]),
            xytext=(change["date"], label_y),
            fontsize=8, fontweight="bold", color=marker_color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=marker_color, alpha=0.8),
            zorder=6,
        )

    ax_chart.set_ylim(0, 105)
    ax_chart.set_ylabel("Score", fontsize=12)
    ax_chart.set_xlabel("Date", fontsize=12)
    ax_chart.set_title(f"{subject} — {student_name} ({class_name})",
                       fontsize=14, fontweight="bold", pad=15)
    ax_chart.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_chart.grid(True, alpha=0.3, linestyle="--")
    ax_chart.xaxis.set_major_formatter(DateFormatter("%m/%d"))

    # Feedback notes panel
    if ax_notes and changes:
        ax_notes.set_xlim(0, 1)
        ax_notes.set_ylim(0, 1)
        ax_notes.axis("off")

        ax_notes.text(0, 0.95, "Feedback Notes:",
                      fontsize=10, fontweight="bold", va="top", transform=ax_notes.transAxes)

        max_per_col = 5
        n_cols = max(1, (len(changes) + max_per_col - 1) // max_per_col)
        col_width = 1.0 / n_cols

        for i, change in enumerate(changes):
            col = i // max_per_col
            row = i % max_per_col
            x_pos = col * col_width + 0.01
            y_pos = 0.80 - row * 0.17

            feedback = generate_feedback(change)
            marker_color = "#27ae60" if change["direction"] == "up" else "#e74c3c"
            date_str = change["date"].strftime("%m/%d")

            note_text = f"{i+1}. [{date_str}] {feedback}"
            ax_notes.text(x_pos, y_pos, note_text, fontsize=9, va="top",
                          color="#333333", transform=ax_notes.transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                    edgecolor=marker_color, alpha=0.7))

    # Save
    safe_subj = subject.replace(" ", "_")
    filepath = output_dir / f"{safe_filename(student_name)}_{safe_subj}_final.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath, len(changes)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR,
                        help=f"Directory containing one CSV per class (default: {DEFAULT_CSV_DIR})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to write charts (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"Glob for CSV files inside --csv-dir (default: {DEFAULT_PATTERN})")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.out_dir

    font = configure_fonts()
    if font is None:
        print("Note: no Hangul-capable font found; Korean characters may render as boxes.")

    print("Loading data...")
    df = load_all_data(args.csv_dir, args.pattern)

    # Remove reference rows (names starting with "*")
    df = df[~df["Name"].str.startswith("*")]
    df = df.dropna(subset=SUBJECTS, how="all")

    # Find active students
    active_students = get_active_students(df)
    print(f"Found {len(active_students)} active students across {len(df['Class'].unique())} classes.")
    print(f"Change threshold: {CHANGE_THRESHOLD} points\n")

    sorted_students = sorted(active_students.items(), key=lambda x: (x[1], x[0]))

    chart_count = 0
    total_changes = 0

    for student_name, current_class in sorted_students:
        # Create class subfolder
        class_dir = output_dir / current_class
        class_dir.mkdir(parents=True, exist_ok=True)

        # Create student subfolder
        student_dir = class_dir / safe_filename(student_name)
        student_dir.mkdir(exist_ok=True)

        # Get ALL data across all classes
        student_data = get_student_all_data(df, student_name)

        subj_count = 0
        student_changes = 0
        for subject in SUBJECTS:
            result = create_annotated_subject_chart(student_name, student_data, subject, current_class, student_dir)
            if result:
                _, n_changes = result
                subj_count += 1
                student_changes += n_changes
                chart_count += 1
                total_changes += n_changes

        if subj_count > 0:
            classes_seen = student_data["Class"].unique().tolist()
            transfer_note = f" (was in: {', '.join(classes_seen)})" if len(classes_seen) > 1 else ""
            print(f"  [{current_class}] {student_name}: {subj_count} charts, {student_changes} changes{transfer_note}")

    print(f"\nDone! {chart_count} annotated charts saved to '{output_dir}/'")
    print(f"Total significant changes detected: {total_changes}")
    print(f"Organized as: {output_dir}/<class>/<student>/<subject>_final.png")


if __name__ == "__main__":
    main()
