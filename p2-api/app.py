"""Project 2 (Deep Learning Applications), Week 8: Brain-Tumour MRI Screening API.

Serves my Week 6 model (ResNet-18, no_tumor vs tumor) and my Week 7 text evidence
(masked clinical note, regex fields, urgent terms, optional NER).

Author: Muhammad Umar Khan | Introduction to Applied AI (BS 8th Semester, Electronics)
Endpoints: GET /health, POST /predict (image), POST /analyze (image + note).
Decision support only; a qualified human decides.
"""
import io
import os
import re
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Settings (change here, or set as environment variables)
# --------------------------------------------------------------------------
MODEL_PATH = os.getenv('MODEL_PATH', 'week6_model.pt')
POSITIVE = os.getenv('POSITIVE', 'tumor')            # the class you must not miss
# Week 6 operating point: chosen on VALIDATION for recall >= 0.95 (test sensitivity 0.972).
# It is NOT stored in the checkpoint. Set THRESHOLD=0.5 for the plain argmax rule.
THRESHOLD = float(os.getenv('THRESHOLD', '0.986'))
API_KEY = os.getenv('API_KEY')                       # set it when you deploy
MAX_BYTES = 5 * 1024 * 1024                          # reject uploads over 5 MB

# Warning terms from the brain-tumour domain (Week 7), matched at the START of a word
URGENT = ['headache', 'seizure', 'vomit', 'blurred vision', 'vision loss',
          'weakness', 'numbness', 'confusion', 'drowsy', 'unconscious',
          'slurred speech', 'memory loss', 'loss of balance']

# Week 7 Part 4 patterns (lab patterns + my brain-tumour MRI domain patterns)
PATTERNS = {
    'cnic':    r'\b\d{5}-\d{7}-\d\b',
    'mobile':  r'(?:\+92[- ]?|\b0)3\d{2}[- ]?\d{7}\b',
    'amount':  r'\b(?:PKR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)',   # \b fixes "Doctors 500 mg"
    'date':    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
    'email':   r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b',
    'invoice': r'\b(?:Invoice\s+)?No\.?\s*(\d{3,})\b',
    # my domain
    'scan_file':   r'\b(?:Tr|Te)-(?:gl|me|no|pi)_\d+\.(?:jpe?g|png)\b',
    'tumour_size': r'\b\d+(?:\.\d+)?\s?(?:mm|cm)\b',
    'mrn':         r'\bMRN[- ]?\d{6,8}\b',
    'age':         r'\baged?\s+(\d{1,2}|1[01]\d)\b',
    'sequence':    r'(?-i:\b(?:T1|T2|FLAIR|DWI)\b)',   # case-SENSITIVE: "flair" the word is not a sequence
}

state = {}


# --------------------------------------------------------------------------
# Optional spaCy domain pipeline (Week 7 Part 3). The API works without it.
#   pip install spacy && python -m spacy download en_core_web_sm
# --------------------------------------------------------------------------
HOSPITAL_SUFFIX = ['hospital', 'society', 'centre', 'center', 'clinic',
                   'institute', 'bank']
TITLES = {'dr', 'mr', 'mrs', 'ms', 'prof', 'patient', 'farmer'}
DOMAIN_RULES = [
    {'label': 'MONEY', 'pattern': [{'LOWER': {'IN': ['pkr', 'rs', 'rs.']}},
                                   {'IS_PUNCT': True, 'OP': '?'},
                                   {'LIKE_NUM': True}]},
    {'label': 'DATE',
     'pattern': [{'TEXT': {'REGEX': r'^\d{1,2}/\d{1,2}/\d{4}$'}}]},
    {'label': 'ORG', 'pattern': 'PIMS'},
    {'label': 'ORG', 'pattern': [
        {'IS_TITLE': True, 'LOWER': {'NOT_IN': ['the', 'dr.']}},
        {'IS_TITLE': True, 'OP': '*'},
        {'LOWER': {'IN': HOSPITAL_SUFFIX}}]},
] + [{'label': 'GPE', 'pattern': c}
     for c in ['Islamabad', 'Rawalpindi', 'Lahore', 'Karachi']]


def build_nlp():
    """Return the Week 7 domain pipeline, or None if spaCy / its model is missing."""
    try:
        import spacy
        from spacy.language import Language
        from spacy.tokens import Span

        if not Language.has_factory('strip_titles'):
            @Language.component('strip_titles')
            def strip_titles(doc):
                ents = []
                for ent in doc.ents:
                    if (ent.label_ == 'PERSON' and len(ent) > 1
                            and ent[0].lower_.rstrip('.') in TITLES):
                        ents.append(Span(doc, ent.start + 1, ent.end, label='PERSON'))
                    else:
                        ents.append(ent)
                doc.ents = ents
                return doc

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('entity_ruler', before='ner').add_patterns(DOMAIN_RULES)
        nlp.add_pipe('strip_titles', after='ner')
        return nlp
    except Exception as exc:                       # spaCy not installed / model missing
        print(f'[info] spaCy entities disabled: {exc}')
        return None


# --------------------------------------------------------------------------
# Model loading (once, at server start)
# --------------------------------------------------------------------------
def rebuild(ck):
    name, k = ck['backbone'], len(ck['classes'])
    if name.startswith('resnet'):
        m = getattr(models, name)(weights=None)
        m.fc = nn.Linear(m.fc.in_features, k)
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, k)
    m.load_state_dict(ck['state_dict'])
    return m.eval()


@asynccontextmanager
async def lifespan(app):
    ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if POSITIVE not in ck['classes']:
        raise RuntimeError(f'POSITIVE={POSITIVE!r} is not one of {ck["classes"]}')
    state['ck'], state['model'] = ck, rebuild(ck)          # load ONCE
    state['tf'] = transforms.Compose(
        ([transforms.Grayscale(3)] if ck['grayscale'] else []) +
        [transforms.Resize((ck['img_size'],) * 2), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
    state['nlp'] = build_nlp()                              # optional
    yield
    state.clear()


DESCRIPTION = """
## Brain-Tumour MRI Screening (No Tumour Vs Tumour)

A ResNet-18 model pretrained on ImageNet (Week 6), combined with clinical-note evidence (Week 7).

* **Predict:** `/predict` takes one MRI slice and returns the label, the probabilities and the latency.
  The positive class `tumor` is flagged when P(tumor) >= 0.986 (the Week 6 operating point,
  with validation recall >= 0.95, test sensitivity 0.972 and specificity 0.965).
* **Analyze:** `/analyze` takes an MRI slice and a clinical note. It returns the masked note
  (CNIC, phone, e-mail, MRN), the extracted fields, the urgent terms, and a REFER or ROUTINE
  decision with reasons.

**Intended Use:** Triage, to help prioritise a radiologist's queue.

**Not For:** Diagnosis, treatment decisions, use without radiologist review, children or
populations that are not in the data, or scans from other scanners (no external validation).

**Limits:** Trained on one public Kaggle dataset with no patient ids and a small healthy test
group (57 images). The test-set PPV (0.997) is not the screening PPV (about 0.59 at 5% prevalence).
"""

TAGS = [
    {'name': 'Health', 'description': 'Check that the server is running and the model is loaded.'},
    {'name': 'Prediction', 'description': 'Send one MRI slice and get the tumour probability.'},
    {'name': 'Analysis', 'description': 'Send an MRI slice with a clinical note and get a masked case record.'},
]

app = FastAPI(title='Project 2: Brain-Tumour MRI Screening API',
              description=DESCRIPTION, version='1.0', lifespan=lifespan,
              openapi_tags=TAGS, docs_url=None)      # /docs is served by custom_docs() below

# Colourful theme for the interactive docs page (injected into the Swagger UI page)
DOCS_HEAD = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 100 100%27><text y=%27.9em%27 font-size=%2790%27>&#129504;</text></svg>">
<style>
  body { background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 50%, #ecfeff 100%); min-height: 100vh; margin: 0; }
  .swagger-ui { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  .swagger-ui .wrapper { max-width: 1100px; }

  /* Header card */
  .swagger-ui .info { margin: 28px 0; padding: 26px 30px; border-radius: 18px; color: #fff;
      background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #ec4899 100%);
      box-shadow: 0 12px 30px rgba(79, 70, 229, .35); }
  .swagger-ui .info .title, .swagger-ui .info p, .swagger-ui .info li, .swagger-ui .info h1,
  .swagger-ui .info h2, .swagger-ui .info h3, .swagger-ui .info strong, .swagger-ui .info table,
  .swagger-ui .info .base-url { color: #fff; }
  .swagger-ui .info .title { font-size: 2.2rem; font-weight: 800; }
  .swagger-ui .info h2 { margin: 18px 0 8px; font-size: 1.5rem; color: #fde68a; }
  .swagger-ui .info a { color: #fde68a; }
  .swagger-ui .info .title small { background: #22c55e; }
  .swagger-ui .info .title small.version-stamp { background: #0ea5e9; }
  .swagger-ui .info .title small pre { color: #fff; }
  .swagger-ui .renderedMarkdown code, .swagger-ui .markdown code { background: rgba(255,255,255,.22);
      color: #fff; border-radius: 6px; padding: 2px 7px; }

  /* Tag headers: one colour per group */
  .swagger-ui .opblock-tag { border-radius: 12px; padding: 12px 18px; margin: 12px 0 6px; color: #fff;
      border-bottom: none; box-shadow: 0 4px 12px rgba(0,0,0,.12); }
  .swagger-ui .opblock-tag small { color: rgba(255,255,255,.9); }
  .swagger-ui .opblock-tag svg { fill: #fff; }
  .swagger-ui .opblock-tag[data-tag="Health"]     { background: linear-gradient(90deg, #0d9488, #14b8a6); }
  .swagger-ui .opblock-tag[data-tag="Prediction"] { background: linear-gradient(90deg, #7c3aed, #a855f7); }
  .swagger-ui .opblock-tag[data-tag="Analysis"]   { background: linear-gradient(90deg, #ea580c, #f59e0b); }

  /* Endpoint boxes: colour follows the group */
  .swagger-ui .opblock { border-radius: 12px; border-width: 2px; box-shadow: 0 4px 12px rgba(0,0,0,.08); margin: 0 0 14px; }
  .swagger-ui .opblock .opblock-summary-method { border-radius: 8px; font-weight: 700; min-width: 80px; }
  .swagger-ui .opblock .opblock-summary-description { font-weight: 600; }
  .swagger-ui .opblock-tag-section:has([data-tag="Health"]) .opblock { background: rgba(20,184,166,.10); border-color: #14b8a6; }
  .swagger-ui .opblock-tag-section:has([data-tag="Health"]) .opblock-summary-method { background: #0d9488; }
  .swagger-ui .opblock-tag-section:has([data-tag="Prediction"]) .opblock { background: rgba(168,85,247,.10); border-color: #a855f7; }
  .swagger-ui .opblock-tag-section:has([data-tag="Prediction"]) .opblock-summary-method { background: #7c3aed; }
  .swagger-ui .opblock-tag-section:has([data-tag="Analysis"]) .opblock { background: rgba(245,158,11,.12); border-color: #f59e0b; }
  .swagger-ui .opblock-tag-section:has([data-tag="Analysis"]) .opblock-summary-method { background: #ea580c; }
  .swagger-ui .opblock .opblock-section-header { background: rgba(255,255,255,.7); }

  /* Buttons */
  .swagger-ui .btn { border-radius: 8px; font-weight: 700; }
  .swagger-ui .btn.try-out__btn { border-color: #7c3aed; color: #7c3aed; }
  .swagger-ui .btn.execute { background: linear-gradient(90deg, #f97316, #ec4899); border: none; color: #fff; }
  .swagger-ui .btn.cancel { border-color: #ef4444; color: #ef4444; }
  .swagger-ui .btn.btn-clear { border-color: #0ea5e9; color: #0ea5e9; }
  .swagger-ui .responses-inner h4, .swagger-ui .responses-inner h5 { color: #4f46e5; }
  .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #4f46e5; }
  .swagger-ui section.models { border: 2px solid #c4b5fd; border-radius: 12px; background: rgba(255,255,255,.7); }
  .swagger-ui section.models h4 { color: #4f46e5; }
</style>
"""


@app.get('/docs', include_in_schema=False)
def custom_docs():
    page = get_swagger_ui_html(
        openapi_url=app.openapi_url, title='Brain-Tumour MRI Screening API: Docs',
        swagger_ui_parameters={'displayRequestDuration': True, 'tryItOutEnabled': True,
                               'docExpansion': 'list', 'defaultModelsExpandDepth': 0})
    html = page.body.decode('utf-8').replace('</head>', DOCS_HEAD + '</head>', 1)
    return HTMLResponse(html)


# --------------------------------------------------------------------------
# Response schemas
# --------------------------------------------------------------------------
class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    latency_ms: float
    threshold: float
    positive_flag: bool          # True when P(POSITIVE) >= threshold


class CaseRecord(BaseModel):
    prediction: Prediction
    note_masked: str
    entities: list[list[str]]    # [text, label]; empty if spaCy is not installed
    fields: dict[str, list[str]]
    urgent_terms: list[str]
    decision: str
    reasons: list[str]
    flags: list[str]
    disclaimer: str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def check_key(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, 'Invalid or missing API key')


async def read_image(file: UploadFile):
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(415, 'Upload a JPEG or PNG image')
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image larger than 5 MB')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, 'File is not a readable image')


def classify(img):
    ck, t0 = state['ck'], time.perf_counter()
    x = state['tf'](img.convert('L' if ck['grayscale'] else 'RGB'))
    with torch.inference_mode():
        p = F.softmax(state['model'](x.unsqueeze(0)), dim=1)[0].tolist()
    raw = dict(zip(ck['classes'], p))
    flagged = raw[POSITIVE] >= THRESHOLD                    # operating-point rule
    if len(raw) == 2:
        other = next(c for c in raw if c != POSITIVE)
        label = POSITIVE if flagged else other
    else:
        label = max(raw, key=raw.get)
    return Prediction(label=label,
                      probabilities={c: round(v, 4) for c, v in raw.items()},
                      latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                      threshold=THRESHOLD, positive_flag=flagged)


def mask_pii(text):
    """Hide personal identifiers BEFORE any extraction or logging."""
    text = re.sub(PATTERNS['cnic'], '[CNIC]', text)
    text = re.sub(PATTERNS['mobile'], '[PHONE]', text)
    text = re.sub(PATTERNS['email'], '[EMAIL]', text)
    text = re.sub(PATTERNS['mrn'], '[MRN]', text, flags=re.IGNORECASE)
    return text


def extract_fields(text):
    out = {}
    for name, pat in PATTERNS.items():
        hits = re.findall(pat, text, flags=re.IGNORECASE)
        if hits:
            out[name] = hits
    return out


def urgent_terms_in(text):
    low = text.lower()
    return [k for k in URGENT if re.search(r'\b' + re.escape(k), low)]


def find_entities(text):
    nlp = state.get('nlp')
    if nlp is None:
        return []
    out = []
    for e in nlp(text).ents:
        if '[' in e.text or e.text.upper() in ('CNIC', 'MRN'):   # skip mask placeholders
            continue
        out.append([e.text, e.label_])
    return out


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse('/docs')       # open the interactive page


@app.get('/health', tags=['Health'], summary='Health Check',
         description='Returns the server status, the model settings and the operating threshold.')
def health():
    ck = state.get('ck', {})
    return {'status': 'ok', 'model_loaded': 'model' in state,
            'classes': ck.get('classes', []), 'positive': POSITIVE,
            'threshold': THRESHOLD, 'backbone': ck.get('backbone'),
            'img_size': ck.get('img_size'), 'grayscale': ck.get('grayscale'),
            'entities_enabled': state.get('nlp') is not None}


@app.post('/predict', response_model=Prediction, tags=['Prediction'],
          summary='Predict From MRI Image',
          description='Upload one JPEG or PNG MRI slice (up to 5 MB). Returns the label, the probabilities and the latency.')
async def predict(file: UploadFile = File(...),
                  x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    img = await read_image(file)
    return await run_in_threadpool(classify, img)          # keep the loop free


@app.post('/analyze', response_model=CaseRecord, tags=['Analysis'],
          summary='Analyze MRI Image And Clinical Note',
          description='Upload an MRI slice and a clinical note (up to 2,000 characters). Returns a masked case record with a REFER or ROUTINE decision.')
async def analyze(file: UploadFile = File(...), note: str = Form(...),
                  x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    if len(note) > 2000:
        raise HTTPException(422, 'Note longer than 2,000 characters')
    pred = await run_in_threadpool(classify, await read_image(file))
    safe = mask_pii(note)                                  # mask FIRST
    entities = await run_in_threadpool(find_entities, safe)
    fields = extract_fields(safe)
    urgent = urgent_terms_in(safe)
    reasons = []
    if pred.positive_flag:
        reasons.append(f'image: P({POSITIVE}) = '
                       f'{pred.probabilities[POSITIVE]:.3f}')
    if urgent:
        reasons.append('note mentions: ' + ', '.join(urgent))
    flags = []
    if pred.positive_flag != bool(urgent):
        flags.append('IMAGE/NOTE DISAGREE: send to a human reviewer')
    return CaseRecord(
        prediction=pred, note_masked=safe, entities=entities, fields=fields,
        urgent_terms=urgent, decision='REFER' if reasons else 'ROUTINE',
        reasons=reasons, flags=flags,
        disclaimer='Decision support only; a qualified human decides.')
