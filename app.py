from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import sys
import pandas as pd
import os
import shutil

app = Flask(__name__)
app.secret_key = "attendance_secret_key"

# ---------------- HOME / ROLE ----------------
@app.route("/")
def role():
    return render_template("role.html")

# ---------------- TEACHER PANEL ----------------
@app.route("/teacher")
def teacher():
    return render_template("teacher.html")

# ---------------- REGISTER STUDENT ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student_id = request.form["id"]
        student_name = request.form["name"]

        if not os.path.exists("students.csv"):
            pd.DataFrame(columns=["ID", "Name"]).to_csv("students.csv", index=False)

        df = pd.read_csv("students.csv")
        df = pd.concat(
            [df, pd.DataFrame([[student_id, student_name]], columns=["ID", "Name"])],
            ignore_index=True
        )
        df.to_csv("students.csv", index=False)

        subprocess.run([sys.executable, "capture_faces.py", student_id])

        flash("✅ Student registration completed successfully")
        return redirect(url_for("teacher"))

    return render_template("register.html")

# ---------------- TRAIN MODEL ----------------
@app.route("/train")
def train():
    subprocess.run([sys.executable, "train_model.py"])
    flash("✅ Model training completed successfully")
    return redirect(url_for("teacher"))

# ---------------- SHOW REGISTERED STUDENTS ----------------
@app.route("/students")
def students():
    if os.path.exists("students.csv"):
        df = pd.read_csv("students.csv")
        students = df.to_dict(orient="records")
    else:
        students = []
    return render_template("registered_students.html", students=students)

# ---------------- REMOVE STUDENT ----------------
@app.route("/remove/<student_id>")
def remove_student(student_id):
    if os.path.exists("students.csv"):
        df = pd.read_csv("students.csv")
        df = df[df["ID"].astype(str) != student_id]
        df.to_csv("students.csv", index=False)

    dataset_path = f"dataset/{student_id}"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    flash("❌ Student registration removed successfully")
    return redirect(url_for("students"))

# ---------------- STUDENT PANEL ----------------
@app.route("/student")
def student():
    return render_template("student.html")

# ---------------- MARK ATTENDANCE ----------------
@app.route("/mark", methods=["POST"])
def mark():
    result = subprocess.run(
        [sys.executable, "attendance.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = result.stdout.strip()

    if output.startswith("SUCCESS"):
        try:
            _, sid, name, time = output.split("|")
            flash(
                f"✅ Attendance marked successfully\n"
                f"ID: {sid}\n"
                f"Name: {name}\n"
                f"Time: {time}"
            )
        except:
            flash("❌ Attendance marked, but output error")
    else:
        flash("❌ Attendance not marked")

    return redirect(url_for("student"))

# ---------------- SHOW ATTENDANCE ----------------
@app.route("/attendance")
def show_attendance():
    file = "attendance/attendance.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
        data = df.to_dict(orient="records")
    else:
        data = []
    return render_template("show_attendance.html", data=data)

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
