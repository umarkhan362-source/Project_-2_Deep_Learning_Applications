"""Run with: pytest -q test_api.py

Project 2 (Brain-Tumour MRI Screening API), Week 8: 8 automated tests.
Needs app.py, week6_model.pt and sample.jpeg in the same folder as this file.

Expected values come from my notebooks:
  Week 6: classes ['no_tumor', 'tumor'], resnet18, 224 px, grayscale, threshold 0.986
  Week 7: the case-record note, its masked placeholders, urgent terms and regex fields
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Work from any folder: the model path is resolved next to this file (unless already set)
os.environ.setdefault('MODEL_PATH', str(HERE / 'week6_model.pt'))

import app as api  # noqa: E402  (must come after MODEL_PATH is set)

# ---- values from my Week 6 and Week 7 notebooks ---------------------------------
EXPECTED_CLASSES = ['no_tumor', 'tumor']
EXPECTED_THRESHOLD = 0.986
NOTE = ('Patient Imran Qureshi, aged 46, has had headaches for 3 weeks with vomiting '
        'and one seizure on 10/11/2026. MRI shows a 2.5 cm lesion on T2. Referred from '
        'Mayo Hospital Lahore. MRN 4471823. CNIC 35202-1234567-1, '
        'family mobile 0300-1234567.')
LEAKS = ['35202-1234567-1', '0300-1234567', '35202', '1234567', '4471823']


def find_sample():
    """sample.jpeg next to this file (also accepts sample.jpg / sample.png)."""
    for name in ('sample.jpeg', 'sample.jpg', 'sample.png', 'sample.jpg.jpg'):
        if (HERE / name).exists():
            return HERE / name
    return None


SAMPLE = find_sample()


def ctype_of(path):
    return 'image/png' if Path(path).suffix.lower() == '.png' else 'image/jpeg'


@pytest.fixture(scope='module')
def sample():
    if SAMPLE is None:
        pytest.fail(f'sample.jpeg not found in {HERE}. Copy one brain MRI image there '
                    'and name it sample.jpeg (check with: dir sample*).')
    return SAMPLE


@pytest.fixture(scope='module')
def client():
    """One server start for all tests: the model is loaded once, as in production."""
    model = Path(os.environ['MODEL_PATH'])
    if not model.exists():
        pytest.fail(f'{model} not found. Copy week6_model.pt into {HERE}.')
    with TestClient(api.app) as c:
        yield c


def post_image(client, path, **kw):
    with open(path, 'rb') as f:
        return client.post('/predict', files={'file': ('x', f, ctype_of(path))}, **kw)


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    h = r.json()
    assert h['model_loaded']
    assert h['classes'] == EXPECTED_CLASSES            # Week 6 class order
    assert h['positive'] == 'tumor'
    assert h['backbone'] == 'resnet18' and h['img_size'] == 224 and h['grayscale'] is True
    if 'THRESHOLD' not in os.environ:                  # default = Week 6 operating point
        assert abs(h['threshold'] - EXPECTED_THRESHOLD) < 1e-9


def test_predict_schema_and_probabilities(client, sample):
    r = post_image(client, sample)
    assert r.status_code == 200
    body = r.json()
    assert {'label', 'probabilities', 'latency_ms', 'threshold',
            'positive_flag'} <= set(body)
    probs = body['probabilities']
    assert set(probs) == set(EXPECTED_CLASSES)
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert body['latency_ms'] > 0
    # operating-point rule from Week 6: tumour only when P(tumor) >= threshold
    if abs(probs['tumor'] - body['threshold']) > 1e-3:      # ignore rounding at the edge
        assert body['positive_flag'] == (probs['tumor'] >= body['threshold'])
    assert body['label'] == ('tumor' if body['positive_flag'] else 'no_tumor')


def test_parity_with_direct_model(client, sample):
    served = post_image(client, sample).json()['probabilities']
    ck, tf = api.state['ck'], api.state['tf']
    img = Image.open(sample).convert('L' if ck['grayscale'] else 'RGB')
    with torch.inference_mode():
        p = F.softmax(api.state['model'](tf(img).unsqueeze(0)), 1)[0]
    direct = dict(zip(ck['classes'], p.tolist()))
    for k in direct:
        assert abs(served[k] - direct[k]) < 1e-3       # same numbers


def test_rejects_wrong_type(client):
    r = client.post('/predict', files={'file': ('a.txt', b'hello', 'text/plain')})
    assert r.status_code == 415
    assert r.json()['detail'] == 'Upload a JPEG or PNG image'
    assert 'Traceback' not in r.text                   # no stack trace for the client


def test_rejects_too_large(client):
    big = b'0' * (api.MAX_BYTES + 1)
    r = client.post('/predict', files={'file': ('b.jpg', big, 'image/jpeg')})
    assert r.status_code == 413
    assert r.json()['detail'] == 'Image larger than 5 MB'


def test_rejects_corrupt_image(client):
    r = client.post('/predict', files={'file': ('c.jpg', b'not an image', 'image/jpeg')})
    assert r.status_code == 400
    assert r.json()['detail'] == 'File is not a readable image'


def test_missing_file_is_422(client, sample):
    assert client.post('/predict').status_code == 422                  # no file
    with open(sample, 'rb') as f:                                      # no note
        r = client.post('/analyze', files={'file': ('x', f, ctype_of(sample))})
    assert r.status_code == 422
    with open(sample, 'rb') as f:                                      # note too long
        r = client.post('/analyze', files={'file': ('x', f, ctype_of(sample))},
                        data={'note': 'x' * 2001})
    assert r.status_code == 422
    assert r.json()['detail'] == 'Note longer than 2,000 characters'


def test_analyze_masks_pii(client, sample):
    with open(sample, 'rb') as f:
        r = client.post('/analyze', files={'file': ('x', f, ctype_of(sample))},
                        data={'note': NOTE})
    assert r.status_code == 200
    body = r.json()
    blob = json.dumps(body)
    for leak in LEAKS:                                 # no CNIC / phone / MRN digits anywhere
        assert leak not in blob
    masked = body['note_masked']
    assert '[CNIC]' in masked and '[PHONE]' in masked and '[MRN]' in masked
    # Week 7 notebook values for this note
    assert body['decision'] == 'REFER'
    for term in ('headache', 'seizure', 'vomit'):
        assert term in body['urgent_terms']
    assert body['fields']['tumour_size'] == ['2.5 cm']
    assert body['fields']['age'] == ['46']
    assert body['fields']['sequence'] == ['T2']
    assert body['fields']['date'] == ['10/11/2026']
    assert any(x.startswith('note mentions') for x in body['reasons'])
    assert 'Decision support only' in body['disclaimer']
