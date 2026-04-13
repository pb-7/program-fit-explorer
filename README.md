# Program Fit Explorer

Program Fit Explorer is an LLM-powered decision-support application that evaluates whether a student is a good fit for a graduate program based on their background, interests, and experience.

The app uses a local GGUF-based large language model (LLM) via `llama-cpp-python` to generate realistic “Yes/No” application decisions along with explanations, and provides supporting analytics and visualizations.

---

## 🚀 Features

* Upload a program description (TXT)
* Upload student profiles (CSV)
* Generate application decisions using a local LLM
* Provide first-person explanations for decisions
* Compute:

  * Program alignment scores
  * Model confidence
  * Certainty ranges
* Interactive dashboard with:

  * Decision breakdown
  * Background-type analysis
  * Top/bottom candidate insights

---

## 📂 Project Structure

```
program-fit-explorer/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── models/
│   └── .gitkeep
├── sample_data/
│   ├── product_description.txt
│   └── sample_student_profiles.csv
```

---

## 📥 Input Requirements

### Program Description

* Plain text file (`.txt`)

### Student Profiles CSV

Must contain the following columns:

* `academic_background`
* `academic_interests`
* `professional_interests`
* `previous_work_experience`

---

## ⚙️ Local Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

---

## 🤖 Model Setup

This app uses a local GGUF model (e.g., Phi-3 Mini).

Place your model file inside:

```
models/
```

Example:

```
models/Phi-3-mini-4k-instruct-q4.gguf
```

You can also provide a custom path via the app sidebar.

---

## 📊 Sample Data

Sample files are included in:

```
sample_data/
```

Use them to quickly test the app.

---

## 🧠 Notes

* The app is designed for local LLM inference using `llama-cpp-python`
* Performance depends on your hardware (CPU/GPU)
* Larger datasets may take longer to process

---

## 🚧 Future Improvements

* Hugging Face deployment (Docker-based)
* Model hosting and loading optimization
* Batch processing optimization
* UI improvements and export options

