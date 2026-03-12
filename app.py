from flask import Flask, render_template, request, send_file
import joblib
import re
import PyPDF2
import numpy as np
import difflib
import faiss

from docx import Document
from sentence_transformers import SentenceTransformer

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

app = Flask(__name__)

# =============================
# LOAD MODELS
# =============================

ai_model = joblib.load("ai_detector.pkl")
ai_vectorizer = joblib.load("ai_vectorizer.pkl")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =============================
# IGNORE HEADERS
# =============================

IGNORE_PHRASES = [
"references","bibliography","acknowledgment","acknowledgement",
"introduction","conclusion","abstract","keywords",
"results","method","methods","discussion"
]

# =============================
# REMOVE REFERENCES SECTION
# =============================

def remove_references(text):

    text_lower = text.lower()

    if "references" in text_lower:
        index = text_lower.find("references")
        text = text[:index]

    return text

# =============================
# VALID SENTENCE FILTER
# =============================

def is_valid_sentence(sentence):

    s = sentence.lower().strip()

    if len(s) < 30:
        return False

    digits = sum(c.isdigit() for c in s)

    if digits > len(s)*0.4:
        return False

    for phrase in IGNORE_PHRASES:
        if phrase in s:
            return False

    if re.search(r'\[\d+\]', s):
        return False

    return True

# =============================
# TEXT EXTRACTION
# =============================

def extract_text(file):

    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):

        reader = PyPDF2.PdfReader(file)
        text=""

        for page in reader.pages:
            content=page.extract_text()
            if content:
                text+=content

        return text

    elif filename.endswith(".docx"):

        doc=Document(file)
        text=""

        for para in doc.paragraphs:
            text+=para.text+" "

        return text

    return ""

# =============================
# SPLIT SENTENCES
# =============================

def split_sentences(text):

    sentences = re.split(r'[.!?]+', text)

    sentences = [
        s.strip() for s in sentences
        if is_valid_sentence(s)
    ]

    return sentences

# =============================
# WORD HIGHLIGHTING
# =============================

def highlight_text(user_sentence, ref_sentence):

    user_words=user_sentence.split()
    ref_words=ref_sentence.split()

    matcher=difflib.SequenceMatcher(None,user_words,ref_words)

    hu=[]
    hr=[]

    for opcode,a1,a2,b1,b2 in matcher.get_opcodes():

        if opcode=="equal":

            for w in user_words[a1:a2]:
                hu.append(f'<b>{w}</b>')

            for w in ref_words[b1:b2]:
                hr.append(f'<b>{w}</b>')

        else:

            hu.extend(user_words[a1:a2])
            hr.extend(ref_words[b1:b2])

    return " ".join(hu)," ".join(hr)

# =============================
# FAISS MATCHING
# =============================

def find_matching_sentences(ref_sent,user_sent,ref_sources):

    matches=[]

    ref_emb=embedding_model.encode(ref_sent)
    user_emb=embedding_model.encode(user_sent)

    ref_emb=np.array(ref_emb).astype("float32")
    user_emb=np.array(user_emb).astype("float32")

    dimension=ref_emb.shape[1]

    index=faiss.IndexFlatL2(dimension)
    index.add(ref_emb)

    distances,indices=index.search(user_emb,5)

    for i,user_sentence in enumerate(user_sent):

        best=None
        best_score=0

        for j in range(5):

            ref_idx=indices[i][j]

            sim=1/(1+distances[i][j])
            sim_percent=round(sim*100,2)

            if sim_percent>best_score:

                best_score=sim_percent

                best={
                "user":user_sentence,
                "reference":ref_sent[ref_idx],
                "similarity":sim_percent,
                "source":ref_sources[ref_idx]
                }

        if best and best_score>50:
            matches.append(best)

    return matches

# =============================
# SOURCE CONTRIBUTION
# =============================

def source_statistics(matches):

    stats={}

    for m in matches:

        src=m["source"]

        if src not in stats:
            stats[src]=0

        stats[src]+=1

    return stats

# =============================
# AI DETECTION
# =============================

def ai_percentage(text):

    sentences=split_sentences(text)

    if len(sentences)==0:
        return 0

    ai_count=0

    for s in sentences:

        vec=ai_vectorizer.transform([s])
        pred=ai_model.predict(vec)[0]

        if pred==1:
            ai_count+=1

    percent=(ai_count/len(sentences))*100

    return percent

# =============================
# PDF REPORT
# =============================

def generate_report(similarity,matches,ai_percent,plag_percent,final):

    styles=getSampleStyleSheet()
    elements=[]

    elements.append(Paragraph("Plagiarism and AI Detection Report",styles['Title']))
    elements.append(Spacer(1,20))

    summary=[
    ["Metric","Value"],
    ["Similarity Score",f"{similarity}%"],
    ["Plagiarism Percentage",f"{plag_percent}%"],
    ["AI Probability",f"{round(ai_percent,2)}%"],
    ["Final Result",final]
    ]

    table=Table(summary)

    table.setStyle(TableStyle([
    ('GRID',(0,0),(-1,-1),1,colors.black),
    ('BACKGROUND',(0,0),(-1,0),colors.grey)
    ]))

    elements.append(table)
    elements.append(Spacer(1,20))

    for i,m in enumerate(matches):

        hu,hr=highlight_text(m["user"],m["reference"])

        data=[
        ["User Sentence","Reference Sentence"],
        [Paragraph(hu,styles['Normal']),Paragraph(hr,styles['Normal'])]
        ]

        t=Table(data,colWidths=[250,250])

        t.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)
        ]))

        elements.append(Paragraph(f"Match {i+1} - {m['similarity']}%",styles['Normal']))
        elements.append(t)
        elements.append(Spacer(1,10))

    pdf=SimpleDocTemplate("report.pdf",pagesize=letter)
    pdf.build(elements)

# =============================
# ROUTES
# =============================

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    ref_files=request.files.getlist('reference')
    user_file=request.files['user']

    user_text=extract_text(user_file)
    user_text=remove_references(user_text)

    ref_sent=[]
    ref_sources=[]

    for file in ref_files:

        text=extract_text(file)
        text=remove_references(text)

        sentences=split_sentences(text)

        for s in sentences:
            ref_sent.append(s)
            ref_sources.append(file.filename)

    user_sent=split_sentences(user_text)

    matches=find_matching_sentences(ref_sent,user_sent,ref_sources)

    matched_users=set([m["user"] for m in matches])

    plag_percent=round((len(matched_users)/len(user_sent))*100,2) if user_sent else 0

    similarity_score=round(
    sum(m["similarity"] for m in matches)/len(matches),2
    ) if matches else 0

    ai_percent=ai_percentage(user_text)

    source_stats=source_statistics(matches)

    if plag_percent>25:

        if ai_percent>42:
            final="Plagiarized and AI Generated"
        else:
            final="Plagiarized Human Text"

    else:

        if ai_percent>42:
            final="Original AI Text"
        else:
            final="Original Human Text"

    generate_report(similarity_score,matches,ai_percent,plag_percent,final)

    return render_template(
    "result.html",
    matches=matches,
    sources=source_stats,
    similarity=similarity_score,
    final=final
    )

@app.route('/download')
def download():
    return send_file("report.pdf",as_attachment=True)

if __name__=="__main__":
    app.run(debug=True)