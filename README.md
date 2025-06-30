
# 📚 Subject Classifier

A machine learning-based system to classify educational documents (like past papers, lectures, or notes) into academic subjects such as Computer Science, Chemistry, and more.

---

## 🔍 Overview

This project implements multiple versions of text classification models trained on labeled datasets. It includes preprocessing tools, modular classifiers, and a deployment pipeline with Docker for scalable use.

---

## 🧠 Features

- ✅ Multiple classifier versions (v1–v10) for experimentation and benchmarking
- 🗂️ Cleaned and labeled datasets in CSV and JSON formats
- 🐍 Python-based classifiers using Scikit-learn / Transformers
- 🐳 Docker support for deployment and reproducibility
- 📈 Training-ready pipeline with requirements for quick setup

---

## 🗃️ Folder Structure

```
subject_classifier/
├── classifier versions/              # Different model versions (v1 to v10)
├── datasets used/                   # Input data (CSV and JSON)
├── deployment tools/               # Dockerfile and requirements
├── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ahmedmoharam10/subject_classifier.git
cd subject_classifier
```

### 2. Install Requirements

Create a virtual environment and install:

```bash
pip install -r "deployment tools/requirements_classification.txt"
```

### 3. Run Classifier

Example (using v5):

```bash
python "classifier versions/classifer_v5.py"
```

---

## 🐳 Docker Usage

To build and run using Docker:

```bash
docker build -f "deployment tools/Dockerfile_classification" -t subject_classifier .
docker run subject_classifier
```

---

## 📊 Dataset Format

* **CSV files** contain subject-labeled data.
* **JSON** is used for structured input-output pair training.
* Make sure your text input matches expected fields for each model version.

---

## ✅ TODO
* [ ] Add REST API with FastAPI or Flask

---

## 🤝 Contributing

Pull requests and ideas are welcome. Please fork the repo and open a PR after testing.

---

## 📜 License

This project is under the MIT License — feel free to use, share, and adapt with credit.

---

## 🙌 Author

**Ahmed Moharam**  
[GitHub Profile](https://github.com/ahmedmoharam10)
