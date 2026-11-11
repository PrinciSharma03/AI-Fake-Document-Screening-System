import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------- settings ----------
N_ORIGINALS = 40          # start small; change to 40 once it works
SIZE = (1000, 630)
ROOT = Path(__file__).resolve().parents[2]   # project root folder
GENUINE_DIR = ROOT / "data" / "raw" / "genuine" / "id_card"
FAKE_DIR = ROOT / "data" / "raw" / "manipulated" / "id_card"
LABELS_CSV = ROOT / "data" / "raw" / "labels_id_card.csv"
random.seed(7)

FIRST = ["Aarav", "Meera", "Kabir", "Simran", "Rohan", "Ananya", "Arjun", "Isha", "Vikram", "Tanvi"]
LAST = ["Sharma", "Verma", "Gill", "Mehta", "Kapoor", "Singh", "Nair", "Reddy", "Bose", "Joshi"]
COURSES = ["B.Tech CSE", "B.Sc Physics", "BBA", "B.Com", "BCA"]
COLLEGE = "Northfield Institute of Technology"   # fictional

# field: (x, y, font size, bold) -- text starts at x (left aligned)
LAYOUT = {
    "name":   (330, 230, 34, True),
    "id_no":  (330, 300, 28, False),
    "course": (330, 360, 28, False),
    "dob":    (330, 420, 28, False),
    "valid":  (330, 480, 28, False),
}
DISPLAY = {"name": "{}", "id_no": "ID No: {}", "course": "Course: {}",
           "dob": "DOB: {}", "valid": "Valid till: {}"}
BG = (245, 247, 252)
BG_EDIT = (243, 245, 250)   # slightly different colour, like a real patch


def font(size, bold=False):
    """Load a Windows font; fall back to Pillow's default if missing."""
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def random_values():
    """Make up all the fictional details for one ID card."""
    return {
        "name": f"{random.choice(FIRST)} {random.choice(LAST)}",
        "id_no": f"ST{random.randint(2022, 2026)}-{random.randint(1000, 9999)}",
        "course": random.choice(COURSES),
        "dob": f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-{random.randint(1999, 2006)}",
        "valid": f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-{random.randint(2027, 2030)}",
    }


def draw_id_card(values):
    """Draw one genuine ID card image from the values."""
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SIZE[0], 110], fill=(30, 60, 120))          # header band
    d.text((500, 55), COLLEGE, font=font(36, True), fill=(255, 255, 255), anchor="mm")
    d.text((500, 150), "STUDENT IDENTITY CARD", font=font(26, True), fill=(60, 60, 60), anchor="mm")
    d.rectangle([60, 200, 300, 500], fill=(200, 200, 200))         # photo placeholder
    d.text((180, 350), "PHOTO", font=font(28), fill=(120, 120, 120), anchor="mm")
    for field, (x, y, size, bold) in LAYOUT.items():
        text = DISPLAY[field].format(values[field])
        d.text((x, y), text, font=font(size, bold), fill=(20, 20, 20), anchor="lm")
    d.line([650, 570, 930, 570], fill=(60, 60, 60), width=2)       # signature line
    d.text((790, 595), "Signature", font=font(18), fill=(90, 90, 90), anchor="mm")
    d.rectangle([4, 4, SIZE[0] - 4, SIZE[1] - 4], outline=(30, 60, 120), width=4)
    return img


def tamper(img, field, new_value):
    """Copy the image and replace one field, like a simple forgery."""
    fake = img.copy()
    d = ImageDraw.Draw(fake)
    x, y, size, bold = LAYOUT[field]
    d.rectangle([x - 10, y - 28, x + 620, y + 28], fill=BG_EDIT)   # cover old text
    text = DISPLAY[field].format(new_value)
    d.text((x, y), text, font=font(size - 2, bold), fill=(20, 20, 20), anchor="lm")
    return fake


def main():
    GENUINE_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, N_ORIGINALS + 1):
        group = f"ID{i:03d}"                   # original + its edit share this ID
        values = random_values()
        img = draw_id_card(values)
        genuine_name = f"id_{i:03d}.png"
        img.save(GENUINE_DIR / genuine_name)
        rows.append([genuine_name, "id_card", 0, group])

        field = random.choice(["name", "id_no", "dob", "valid"])
        new_value = random_values()[field]
        while new_value == values[field]:
            new_value = random_values()[field]
        fake_name = f"id_{i:03d}_edit1.png"
        tamper(img, field, new_value).save(FAKE_DIR / fake_name)
        rows.append([fake_name, "id_card", 1, group])

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "doc_type", "label", "group_id"])
        writer.writerows(rows)
    print(f"Done: {N_ORIGINALS} genuine + {N_ORIGINALS} manipulated ID cards")


if __name__ == "__main__":
    main()