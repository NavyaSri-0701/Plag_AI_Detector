from flask import Flask, render_template, request, send_file
import joblib
import re
from sentence_transformers import SentenceTransformer

# Swagger
from flasgger import Swagger

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

app = Flask(__name__)
swagger = Swagger(app)

# =============================
# LOAD MODELS
# =============================

classifier = joblib.load("plagiarism_classifier.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

ai_model = joblib.load("ai_detector.pkl")
ai_vectorizer = joblib.load("ai_vectorizer.pkl")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

# =============================
# FUNCTIONS
# =============================

def split_sentences(text):

    sentences = re.split(r'[.!?]+', text)

    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    return sentences


def calculate_similarity(ref_text, user_text):

    v1 = tfidf_vectorizer.transform([ref_text])
    v2 = tfidf_vectorizer.transform([user_text])

    sim = (v1 * v2.T).toarray()[0][0]

    return round(sim * 100, 2)


def check_continuous_match(ref, user):

    continuous = 0

    for u in user:
        for r in ref:

            v1 = tfidf_vectorizer.transform([u])
            v2 = tfidf_vectorizer.transform([r])

            sim = (v1 * v2.T).toarray()[0][0]

            if sim > 0.85:

                continuous += 1

                if continuous >= 6:
                    return True

            else:
                continuous = 0

    return False


def find_matching_sentences(ref, user):

    matches = []

    for u in user:
        for r in ref:

            v1 = tfidf_vectorizer.transform([u])
            v2 = tfidf_vectorizer.transform([r])

            sim = (v1 * v2.T).toarray()[0][0]

            if sim > 0.70:
                matches.append((u, r, round(sim * 100, 2)))

    return matches


def ai_percentage(text):

    sentences = split_sentences(text)

    ai_count = 0

    for s in sentences:

        vec = ai_vectorizer.transform([s])

        pred = ai_model.predict(vec)[0]

        if pred == 1:
            ai_count += 1

    percent = (ai_count / len(sentences)) * 100

    return percent


def generate_report(similarity, matches, ai_percent, final):

    file_path = "report.pdf"

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Plagiarism and AI Detection Report", styles['Title']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Similarity Score : {similarity} %", styles['Normal']))
    elements.append(Paragraph(f"AI Probability : {round(ai_percent,2)} %", styles['Normal']))
    elements.append(Paragraph(f"Final Result : {final}", styles['Normal']))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Matching Sentences", styles['Heading2']))
    elements.append(Spacer(1, 10))

    if len(matches) == 0:

        elements.append(Paragraph("No matching sentences detected.", styles['Normal']))

    else:

        for u, r, sim in matches:

            elements.append(Paragraph(f"<b>User:</b> {u}", styles['Normal']))
            elements.append(Paragraph(f"<b>Reference:</b> {r}", styles['Normal']))
            elements.append(Paragraph(f"Similarity: {sim}%", styles['Normal']))
            elements.append(Spacer(1, 10))

    pdf = SimpleDocTemplate(file_path, pagesize=letter)

    pdf.build(elements)

    return file_path


# =============================
# ROUTES
# =============================

@app.route('/')
def home():
    """
    Home Page
    ---
    responses:
      200:
        description: Web interface
    """
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Plagiarism + AI Detection
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: reference
        in: formData
        type: file
        required: true
        description: Reference document(s)

      - name: user
        in: formData
        type: file
        required: true
        description: User document

    responses:
      200:
        description: Detection results
    """

    ref_files = request.files.getlist('reference')

    user_file = request.files['user']

    user_text = user_file.read().decode('utf-8')

    ref_texts = []

    for file in ref_files:

        text = file.read().decode('utf-8')

        ref_texts.append(text)

    combined_ref_text = " ".join(ref_texts)

    ref_sent = split_sentences(combined_ref_text)

    user_sent = split_sentences(user_text)

    plag = check_continuous_match(ref_sent, user_sent)

    similarity_score = calculate_similarity(combined_ref_text, user_text)

    matches = find_matching_sentences(ref_sent, user_sent)

    ai_percent = ai_percentage(user_text)

    if plag:
        plag_result = "Plagiarized"
    else:
        plag_result = "Not Plagiarized"

    if ai_percent >= 42:
        ai_result = "AI Generated"
    else:
        ai_result = "Human Written"

    if plag and ai_percent >= 42:
        final = "Plagiarized and Also AI Generated"

    elif plag and ai_percent < 42:
        final = "Plagiarized Human Text not AI Generated"

    elif not plag and ai_percent >= 42:
        final = "AI Generated and Not Plagiarised"

    else:
        final = "Original Human Text"

    generate_report(
        similarity_score,
        matches,
        ai_percent,
        final
    )

    return render_template(
        "index.html",
        plag=plag_result,
        ai=ai_result,
        percent=round(ai_percent, 2),
        similarity=similarity_score,
        final=final,
        report=True
    )


@app.route('/download')
def download():
    """
    Download Report
    ---
    responses:
      200:
        description: Download PDF report
    """
    return send_file("report.pdf", as_attachment=True)


# =============================
# RUN
# =============================

if __name__ == '__main__':
    app.run(debug=True)