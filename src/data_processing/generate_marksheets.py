import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------- settings ----------
N_ORIGINALS = 60           # start small; change to 60 once it works
SIZE = (1000, 1300)
ROOT = Path(__file__).resolve().parents[2]   # project root folder
GENUINE_DIR = ROOT / "data" / "raw" / "genuine" / "marksheet"
FAKE_DIR = ROOT / "data" / "raw" / "manipulated" / "marksheet"
LABELS_CSV = ROOT / "data" / "raw" / "labels_marksheet.csv"
random.seed(11)

FIRST = ["Aarav", "Meera", "Kabir", "Simran", "Rohan", "Ananya", "Arjun", "Isha", "Vikram", "Tanvi"]
LAST = ["Sharma", "Verma", "Gill", "Mehta", "Kapoor", "Singh", "Nair", "Reddy", "Bose", "Joshi"]
SUBJECTS = ["Mathematics", "Physics", "Data Structures", "Database Systems",
            "Operating Systems", "Computer Networks", "Software Engineering", "Discrete Mathematics"]
UNIVERSITY = "Northfield University"     # fictional

BG = (248, 248, 244)
BG_EDIT = BG              # slightly different colour, like a real patch
INK = (20, 20, 20)
ROW_Y = [490, 550, 610, 670, 730]        # y position of the 5 subject rows
MARKS_X = 700

# editable field: (x, y, font size, bold, width of area covered when editing)
EDIT = {
    "name": (80, 250, 30, False, 600),
    "roll_no": (80, 300, 30, False, 600),
    "total": (MARKS_X, 810, 30, True, 110),
}
for i, y in enumerate(ROW_Y):
    EDIT[f"marks_{i + 1}"] = (MARKS_X, y, 28, False, 110)

DISPLAY = {"name": "Name: {}", "roll_no": "Roll No: {}", "total": "{}"}
for i in range(5):
    DISPLAY[f"marks_{i + 1}"] = "{}"


def font(size, bold=False):
    """Load a Windows font; fall back to Pillow's default if missing."""
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def grade(m):
    for cut, g in [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (50, "C")]:
        if m >= cut:
            return g
    return "D"


def random_values():
    """Make up all the fictional details for one marksheet."""
    return {
        "name": f"{random.choice(FIRST)} {random.choice(LAST)}",
        "roll_no": f"{random.randint(2021, 2025)}{random.randint(10000, 99999)}",
        "semester": random.randint(1, 8),
        "subjects": random.sample(SUBJECTS, 5),
        "marks": [random.randint(45, 98) for _ in range(5)],
    }


def field_value(values, field):
    """The current value of an editable field."""
    if field == "total":
        return sum(values["marks"])
    if field.startswith("marks_"):
        return values["marks"][int(field.split("_")[1]) - 1]
    return values[field]


def draw_marksheet(values):
    """Draw one genuine marksheet."""
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE[0], 120], fill=(30, 60, 120))
    d.text((500, 60), UNIVERSITY, font=font(42, True), fill=(255, 255, 255), anchor="mm")
    d.text((500, 175), "STATEMENT OF MARKS", font=font(32, True), fill=(60, 60, 60), anchor="mm")

    for field in ("name", "roll_no"):
        x, y, size, bold, _ = EDIT[field]
        d.text((x, y), DISPLAY[field].format(field_value(values, field)),
               font=font(size, bold), fill=INK, anchor="lm")
    d.text((80, 350), f"Program: B.Tech Computer Science     Semester: {values['semester']}",
           font=font(26), fill=INK, anchor="lm")

    d.rectangle([60, 400, 940, 450], fill=(220, 226, 240))          # table header
    for x, t in [(80, "Subject"), (560, "Max"), (MARKS_X, "Marks"), (840, "Grade")]:
        d.text((x, 425), t, font=font(26, True), fill=INK, anchor="lm")

    for i, y in enumerate(ROW_Y):
        m = values["marks"][i]
        d.text((80, y), values["subjects"][i], font=font(28), fill=INK, anchor="lm")
        d.text((560, y), "100", font=font(28), fill=INK, anchor="lm")
        d.text((MARKS_X, y), str(m), font=font(28), fill=INK, anchor="lm")
        d.text((840, y), grade(m), font=font(28), fill=INK, anchor="lm")
        d.line([60, y + 30, 940, y + 30], fill=(200, 200, 200), width=1)

    total = sum(values["marks"])
    d.text((80, 810), "Total", font=font(30, True), fill=INK, anchor="lm")
    d.text((560, 810), "500", font=font(30, True), fill=INK, anchor="lm")
    d.text((MARKS_X, 810), str(total), font=font(30, True), fill=INK, anchor="lm")
    d.text((80, 880), f"Percentage: {total / 5:.1f}%", font=font(28), fill=INK, anchor="lm")
    d.text((80, 930), "Result: PASS", font=font(28, True), fill=INK, anchor="lm")
    d.text((80, 1200), "Issued on: 15-07-2026", font=font(24), fill=INK, anchor="lm")
    d.line([650, 1190, 930, 1190], fill=(60, 60, 60), width=2)
    d.text((790, 1220), "Controller of Examinations", font=font(20), fill=(90, 90, 90), anchor="mm")
    d.rectangle([6, 6, SIZE[0] - 6, SIZE[1] - 6], outline=(30, 60, 120), width=4)
    return img


def tamper(img, field, new_value):
    """Copy the image and replace one field, like a simple forgery."""
    fake = img.copy()
    d = ImageDraw.Draw(fake)
    x, y, size, bold, width = EDIT[field]
    d.rectangle([x - 10, y - 28, x + width, y + 28], fill=BG_EDIT)     # cover old text
    d.text((x, y), DISPLAY[field].format(new_value), font=font(size, bold), fill=INK, anchor="lm")
    return fake


def new_field_value(values, field):
    """A different value for the same field."""
    while True:
        new = field_value(random_values(), field)
        if new != field_value(values, field):
            return new


def main():
    GENUINE_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, N_ORIGINALS + 1):
        group = f"MS{i:03d}"                    # original + its edit share this ID
        values = random_values()
        img = draw_marksheet(values)
        genuine_name = f"marksheet_{i:03d}.png"
        img.save(GENUINE_DIR / genuine_name)
        rows.append([genuine_name, "marksheet", 0, group])

        field = random.choice(list(EDIT.keys()))
        fake_name = f"marksheet_{i:03d}_edit1.png"
        tamper(img, field, new_field_value(values, field)).save(FAKE_DIR / fake_name)
        rows.append([fake_name, "marksheet", 1, group])

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "doc_type", "label", "group_id"])
        writer.writerows(rows)
    print(f"Done: {N_ORIGINALS} genuine + {N_ORIGINALS} manipulated marksheets")


if __name__ == "__main__":
    main()