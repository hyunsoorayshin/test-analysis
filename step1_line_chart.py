"""
Step 1: Generate best-fit line charts — one image per student with a separate subplot per subject.
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

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = BASE_DIR / "sample_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "charts"
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


def safe_filename(student_name):
    return student_name.replace(" ", "_").replace("*", "").replace("/", "-")


def plot_student(student_name, student_data, class_name, output_dir):
    dates = student_data["Date"]
    dates_numeric = (dates - dates.min()).dt.total_seconds() / 86400

    # Find which subjects have enough data
    active_subjects = []
    for subj in SUBJECTS:
        if subj in student_data.columns:
            vals = student_data[subj].values
            if np.nansum(~np.isnan(vals)) >= 2:
                active_subjects.append(subj)

    if not active_subjects:
        return None

    n_subj = len(active_subjects)
    n_cols = 3
    n_rows = (n_subj + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows + 1))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(f"Student Growth Over Time: {student_name} ({class_name})",
                 fontsize=16, fontweight="bold", y=0.98)

    # Flatten axes for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, subj in enumerate(active_subjects):
        ax = axes[idx]
        ax.set_facecolor("#fafafa")
        vals = student_data[subj].values
        color = COLORS.get(subj, "#333333")
        valid = ~np.isnan(vals)

        # Raw data points
        ax.scatter(dates[valid], vals[valid], color=color, alpha=0.4, s=30, zorder=2)

        # Best-fit line
        x_fit, y_fit = get_best_fit(dates_numeric.values, vals)
        if x_fit is not None:
            fit_dates = dates.min() + pd.to_timedelta(x_fit, unit="D")
            ax.plot(fit_dates, y_fit, color=color, linewidth=2.5, zorder=3)

        ax.set_ylim(0, 105)
        ax.set_title(subj, fontsize=13, fontweight="bold", color=color)
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.set_major_formatter(DateFormatter("%m/%d"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    # Hide unused subplots
    for idx in range(len(active_subjects), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filepath = output_dir / f"{safe_filename(student_name)}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


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

    active_students = get_active_students(df)
    print(f"Found {len(active_students)} active students across {len(df['Class'].unique())} classes.\n")

    sorted_students = sorted(active_students.items(), key=lambda x: (x[1], x[0]))

    count = 0
    for student_name, current_class in sorted_students:
        class_dir = output_dir / current_class
        class_dir.mkdir(parents=True, exist_ok=True)

        student_data = get_student_all_data(df, student_name)
        path = plot_student(student_name, student_data, current_class, class_dir)
        if path:
            count += 1
            classes_seen = student_data["Class"].unique().tolist()
            transfer_note = f" (was in: {', '.join(classes_seen)})" if len(classes_seen) > 1 else ""
            print(f"  [{current_class}] {student_name}{transfer_note}")

    print(f"\nDone! {count} charts saved to '{output_dir}/'")
    print(f"Organized as: {output_dir}/<class>/<student>.png")


if __name__ == "__main__":
    main()
