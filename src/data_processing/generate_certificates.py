import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------- settings ----------
N_ORIGINALS = 40           # start small; change to 40 once it works
SIZE = (1000, 700)
ROOT = Path(__file__).resolve().parents[2]   # project root folder
GENUINE_DIR = ROOT / "data" / "raw" / "genuine" / "certificate"
FAKE_DIR = ROOT / "data" / "raw" / "manipulated" / "certificate"
LABELS_CSV = ROOT / "data" / "raw" / "labels_certificate.csv"
random.seed(42)            # same random numbers every run

FIRST = ["Aarav", "Meera", "Kabir", "Simran", "Rohan", "Ananya", "Arjun", "Isha", "Vikram", "Tanvi"]
LAST = ["Sharma", "Verma", "Gill", "Mehta", "Kapoor", "Singh", "Nair", "Reddy", "Bose", "Joshi"]
COURSES = ["Python Programming", "Data Analysis", "Web Development",
           "Machine Learning Basics", "Cloud Fundamentals"]
GRADES = ["A+", "A", "B+", "B"]
INSTITUTE = "Northfield Institute of Technology"   # fictional

# field: (x, y, font size, bold, half-width of the area to cover when editing)
LAYOUT = {
    "name":   (500, 300, 44, True, 300),
    "course": (500, 400, 32, False, 300),
    "date":   (250, 560, 26, False, 170),
    "grade":  (750, 560, 26, False, 170),
}
DISPLAY = {"name": "{}", "course": "{}", "date": "Date: {}", "grade": "Grade: {}"}


def font(size, bold=False):
    """Load a Windows font; fall back to Pillow's default if missing."""
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def random_values():
    """Make up all the fictional details for one certificate."""
    return {
        "name": f"{random.choice(FIRST)} {random.choice(LAST)}",
        "course": random.choice(COURSES),
        "date": f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-{random.randint(2022, 2026)}",
        "grade": random.choice(GRADES),
        "cert_no": f"CERT-{random.randint(100000, 999999)}",
    }


def draw_certificate(values):
    """Draw a genuine certificate image from the values."""
    img = Image.new("RGB", SIZE, (252, 248, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, SIZE[0] - 20, SIZE[1] - 20], outline=(30, 60, 120), width=6)
    d.text((500, 110), INSTITUTE, font=font(36, True), fill=(30, 60, 120), anchor="mm")
    d.text((500, 190), "Certificate of Completion", font=font(30), fill=(60, 60, 60), anchor="mm")
    d.text((500, 250), "This is to certify that", font=font(22), fill=(80, 80, 80), anchor="mm")
    d.text((500, 350), "has successfully completed", font=font(22), fill=(80, 80, 80), anchor="mm")
    for field, (x, y, size, bold, _) in LAYOUT.items():
        text = DISPLAY[field].format(values[field])
        d.text((x, y), text, font=font(size, bold), fill=(20, 20, 20), anchor="mm")
    d.text((500, 640), values["cert_no"], font=font(20), fill=(90, 90, 90), anchor="mm")
    return img


def tamper(img, field, new_value):
    """Copy the image and replace one field, like a simple forgery."""
    fake = img.copy()
    d = ImageDraw.Draw(fake)
    x, y, size, bold, half = LAYOUT[field]
    d.rectangle([x - half, y - 30, x + half, y + 30], fill=(250, 246, 232))  # cover old text
    text = DISPLAY[field].format(new_value)
    d.text((x, y), text, font=font(size - 2, bold), fill=(20, 20, 20), anchor="mm")
    return fake


def main():
    GENUINE_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, N_ORIGINALS + 1):
        group = f"CERT{i:03d}"                 # original + its edit share this ID
        values = random_values()
        img = draw_certificate(values)
        genuine_name = f"cert_{i:03d}.png"
        img.save(GENUINE_DIR / genuine_name)
        rows.append([genuine_name, "certificate", 0, group])

        field = random.choice(["name", "date", "grade"])
        new_value = random_values()[field]
        while new_value == values[field]:
            new_value = random_values()[field]
        fake_name = f"cert_{i:03d}_edit1.png"
        tamper(img, field, new_value).save(FAKE_DIR / fake_name)
        rows.append([fake_name, "certificate", 1, group])

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "doc_type", "label", "group_id"])
        writer.writerows(rows)
    print(f"Done: {N_ORIGINALS} genuine + {N_ORIGINALS} manipulated images")


if __name__ == "__main__":
    main()