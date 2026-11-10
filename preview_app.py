"""Temporary Flask app for previewing the frontend with demonstration data."""

from flask import Flask, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def preview_result():
    return render_template(
        "result.html",
        filename="sample_marksheet.jpg",
        screening_result="Preview only — no actual analysis performed.",
        warnings=["This is demonstration content, not a real screening result."],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
