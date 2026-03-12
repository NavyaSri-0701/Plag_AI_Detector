# A Classical and Deep Learning-Based NLP System for Plagiarism and AI Text Detection

## Overview

This project presents an intelligent plagiarism detection system that analyzes a user-submitted document and compares it with reference documents to detect textual similarity and determine whether the content is original, plagiarized, or AI-generated.

The system combines **Natural Language Processing (NLP)** techniques, **semantic similarity models**, and **machine learning classifiers** to produce detailed analysis results similar to professional plagiarism tools like Turnitin.

The application provides a **web interface built with Flask**, allowing users to upload documents and view interactive comparison results.

---


## Features

### 1. Plagiarism Detection

* Uses **semantic similarity** between sentences to detect plagiarism.
* Compares the user document with multiple reference documents.
* Calculates similarity using **sentence embeddings**.

### 2. AI-Generated Text Detection

* Uses a trained machine learning model to estimate the probability that text is AI-generated.
* Classifies text as:

  * Original Human Text
  * Original AI Text
  * Plagiarized Human Text
  * Plagiarized and AI Generated

### 3. Interactive Result Visualization

* Color-coded similarity comparison.
* Sentence-level similarity scores.
* Source document contribution statistics.

### 4. Turnitin-Style Word Highlighting

* Matching words between user sentences and reference sentences are highlighted.

### 5. Smart Filtering

The system removes irrelevant sections before analysis:

* References
* Bibliography
* Citations
* Years and numeric-heavy sentences
* Common research paper headers (Abstract, Introduction, etc.)

### 6. PDF Report Generation

Users can download a detailed plagiarism report containing:

* Similarity score
* AI probability
* Plagiarism percentage
* Sentence comparison results

---

## Technologies Used

### Programming Language

* Python

### Web Framework

* Flask

### Machine Learning / NLP

* Sentence Transformers
* TF-IDF Vectorization
* Cosine Similarity
* FAISS (Facebook AI Similarity Search)

### Libraries

* PyPDF2
* python-docx
* scikit-learn
* joblib
* difflib
* NumPy
* ReportLab

### Frontend

* HTML
* CSS
* Jinja2 Templates

---

## System Architecture

1. **Document Upload**

   * User uploads one document.
   * Multiple reference documents are uploaded.

2. **Text Extraction**

   * Text is extracted from PDF, DOCX, or TXT files.

3. **Preprocessing**

   * Remove references and headers
   * Sentence segmentation
   * Filtering irrelevant sentences

4. **Embedding Generation**

   * Sentence embeddings generated using Sentence Transformers.

5. **Similarity Search**

   * FAISS index used for efficient similarity matching.

6. **AI Detection**

   * TF-IDF features passed to trained ML classifier.

7. **Result Generation**

   * Similarity score
   * Plagiarism percentage
   * AI probability
   * Sentence-level comparisons

8. **Report Generation**

   * Results displayed on web interface
   * Downloadable PDF report generated.

---

## Project Structure

```
project/
│
├── app.py
├── ai_detector.pkl
├── ai_vectorizer.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── report.pdf
└── README.md
```

---

## Installation

### Step 1 – Clone the repository

```
git clone https://github.com/your-repository/plagiarism-detector
cd plagiarism-detector
```

### Step 2 – Install dependencies

```
pip install flask
pip install sentence-transformers
pip install scikit-learn
pip install faiss-cpu
pip install PyPDF2
pip install python-docx
pip install reportlab
```

### Step 3 – Run the application

```
python app.py
```

### Step 4 – Open browser

```
http://127.0.0.1:5000
```

---

## Usage

1. Upload **reference documents**.
2. Upload **user document**.
3. Click **Check Plagiarism**.
4. View the results:

   * Similarity score
   * AI probability
   * Plagiarism percentage
   * Sentence comparisons
5. Download the **PDF report**.

---

## Output Example

The system displays:

* Similarity Score
* Plagiarism Percentage
* AI Probability
* Final Classification
* Source Document Contribution
* Sentence-level comparison table

Results are color-coded for easy interpretation.

---

## Advantages

* Detects **semantic plagiarism**, not only exact matches.
* Identifies **AI-generated content**.
* Supports **multiple document formats**.
* Provides **interactive visualization**.

---

## Limitations

* Requires reference documents for comparison.
* AI detection accuracy depends on training data.
* Processing time increases with large datasets.

---

## Future Enhancements

* Deep learning models for improved AI detection
* Large-scale academic database integration
* Real-time plagiarism API
* Document-level similarity visualization
* Multi-language plagiarism detection

---

## Conclusion

This project demonstrates an intelligent approach to plagiarism detection by combining traditional NLP techniques with modern semantic embedding models. The system offers detailed similarity analysis and AI detection, making it useful for academic and research environments.

---
