"""
Generate synthetic sample data for the chart scripts.

Writes one CSV per class to sample_data/ in the same layout the analysis
scripts expect:

    Date,Name,Vocabulary,Grammar HW,Grammar Speaking,Science HW,Science Test,
    Science Textbook,Reading,Grammar Test,Speaking,Novel

- 4 classes (A1..A4), 5-6 students each, names like "StudentA 학생A"
- 10 weekly dates in M/D/YY form (e.g. 1/2/26)
- Scores 40-100 with a few blanks
- A handful of deliberate >=15-point week-over-week jumps so step2 has
  something to annotate
- One "*Reference 기준" row per date per class (filtered out by the scripts)
- One student who transfers from A1 to A2 halfway through, to exercise the
  class-transfer logic

Everything is seeded, so re-running the script reproduces the same files.
Names and scores are entirely made up.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "sample_data"

CLASSES = ["A1", "A2", "A3", "A4"]
STUDENTS_PER_CLASS = {"A1": 6, "A2": 5, "A3": 6, "A4": 5}
NUM_WEEKS = 10
START_DATE = date(2026, 1, 2)

COLUMNS = [
    "Date", "Name",
    "Vocabulary", "Grammar HW", "Grammar Speaking", "Science HW",
    "Science Test", "Science Textbook", "Reading", "Grammar Test",
    "Speaking", "Novel",
]
SCORE_COLUMNS = COLUMNS[2:]

REFERENCE_NAME = "*Reference 기준"
BLANK_PROBABILITY = 0.05

# (student index across all classes, subject, week index, previous score, new score)
# Week index is 0-based; the jump lands between week-1 and week.
DELIBERATE_JUMPS = [
    (0, "Vocabulary", 5, 62, 88),      # +26
    (2, "Reading", 7, 85, 61),         # -24
    (6, "Science Test", 4, 70, 87),    # +17
]

# Student index (across all classes) who moves from A1 to A2 after this week.
TRANSFER_STUDENT_INDEX = 4
TRANSFER_AFTER_WEEK = 4  # weeks 0-4 in A1, weeks 5-9 in A2


def fmt_date(d):
    """M/D/YY without zero padding, matching the input format the scripts parse."""
    return f"{d.month}/{d.day}/{d.year % 100:02d}"


def student_name(index):
    letter = chr(ord("A") + index)
    return f"Student{letter} 학생{letter}"


def build_students():
    """Assign a name and per-week class to every student."""
    students = []
    index = 0
    for cls in CLASSES:
        for _ in range(STUDENTS_PER_CLASS[cls]):
            classes_by_week = [cls] * NUM_WEEKS
            if index == TRANSFER_STUDENT_INDEX:
                assert cls == "A1", "transfer student must start in A1"
                for w in range(TRANSFER_AFTER_WEEK + 1, NUM_WEEKS):
                    classes_by_week[w] = "A2"
            students.append({
                "index": index,
                "name": student_name(index),
                "classes": classes_by_week,
            })
            index += 1
    return students


def generate_scores(rng, students):
    """scores[student_index][week][subject] -> int or None."""
    scores = {}
    for s in students:
        per_week = []
        baselines = {subj: rng.randint(50, 90) for subj in SCORE_COLUMNS}
        for _ in range(NUM_WEEKS):
            row = {}
            for subj in SCORE_COLUMNS:
                if rng.random() < BLANK_PROBABILITY:
                    row[subj] = None
                else:
                    value = baselines[subj] + rng.randint(-7, 7)
                    row[subj] = max(40, min(100, value))
            per_week.append(row)
        scores[s["index"]] = per_week

    for idx, subj, week, prev, new in DELIBERATE_JUMPS:
        scores[idx][week - 1][subj] = prev
        scores[idx][week][subj] = new
    return scores


def main():
    rng = random.Random(SEED)
    students = build_students()
    scores = generate_scores(rng, students)
    dates = [START_DATE + timedelta(weeks=w) for w in range(NUM_WEEKS)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        rows = []
        for w, d in enumerate(dates):
            rows.append([fmt_date(d), REFERENCE_NAME] + [100] * len(SCORE_COLUMNS))
            for s in students:
                if s["classes"][w] != cls:
                    continue
                week_scores = scores[s["index"]][w]
                rows.append(
                    [fmt_date(d), s["name"]]
                    + ["" if week_scores[subj] is None else week_scores[subj] for subj in SCORE_COLUMNS]
                )
        path = OUT_DIR / f"{cls}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"wrote {path.relative_to(OUT_DIR.parent)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
