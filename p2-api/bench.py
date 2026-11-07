"""Latency benchmark against a RUNNING server: python bench.py

Project 2 (Brain-Tumour MRI Screening API), Week 8.

1. Terminal 1 (folder p2-api, venv active):   uvicorn app:app        (no --reload)
2. Terminal 2 (same folder, venv active):     python bench.py
   More requests:  python bench.py 100       Other address:  python bench.py 50 http://127.0.0.1:8001

It prints your hardware, checks the server against my Week 6 / Week 7 notebook values,
measures /predict and /analyze, applies Little's law, runs the bad-input checks
(415, 413, 400, 422) and writes bench_results.txt with a table you can paste in the README.
"""
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
BASE = sys.argv[2].rstrip('/') if len(sys.argv) > 2 else 'http://127.0.0.1:8000'

# ---- values from my Week 6 and Week 7 notebooks ---------------------------------
EXPECTED_CLASSES = ['no_tumor', 'tumor']
EXPECTED_THRESHOLD = 0.986
NOTE = ('Patient Imran Qureshi, aged 46, has had headaches for 3 weeks with vomiting '
        'and one seizure on 10/11/2026. MRI shows a 2.5 cm lesion on T2. Referred from '
        'Mayo Hospital Lahore. MRN 4471823. CNIC 35202-1234567-1, '
        'family mobile 0300-1234567.')
LEAKS = ['35202-1234567-1', '0300-1234567', '35202', '1234567', '4471823']
MAX_BYTES = 5 * 1024 * 1024

lines, checks = [], []

try:                                           # Windows consoles: never crash on a character
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass


def out(text=''):
    print(text)
    lines.append(text)


def check(name, ok, detail=''):
    checks.append((name, bool(ok)))
    out(f'  [{"OK" if ok else "CHECK"}] {name}' + (f'  ({detail})' if detail else ''))


# --------------------------------------------------------------------------
# Sample image (the tests and this file need the exact name sample.jpeg)
# --------------------------------------------------------------------------
def find_sample():
    for name in ('sample.jpeg', 'sample.jpg', 'sample.png', 'sample.jpg.jpg'):
        if (HERE / name).exists():
            return HERE / name
    return None


# --------------------------------------------------------------------------
# Hardware (the lab asks for CPU model and RAM)
# --------------------------------------------------------------------------
def cpu_model():
    try:
        if platform.system() == 'Windows':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown (see Task Manager > Performance)'


def ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    try:
        if platform.system() == 'Windows':
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024 ** 3
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1024 ** 3
    except Exception:
        return None


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def bench(client, name, url, data, ctype, with_note=False):
    """N timed requests (one warm-up request is not counted). Returns latencies in ms."""
    form = {'note': NOTE} if with_note else None

    def call():
        return client.post(url, files={'file': ('x.jpeg', data, ctype)}, data=form)

    call().raise_for_status()                                  # warm-up
    lat = []
    for _ in range(N):
        t0 = time.perf_counter()
        call().raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    p50, p95 = np.percentile(lat, 50), np.percentile(lat, 95)
    out(f'{name:9s} n={N}  p50={p50:.0f} ms  p95={p95:.0f} ms  '
        f'max={lat.max():.0f} ms  mean={lat.mean():.0f} ms')
    out(f'          Throughput with 1 worker ~ {1000 / lat.mean():.1f} requests/s')
    if p95 > 3 * p50:
        out('          Note: p95 is more than 3x p50 (heavy tail). Close other programs and '
            'run uvicorn without --reload.')
    return lat


def littles_law(lat, name):
    """L = lambda * W. One worker (L = 1) handles 1 / W requests per second."""
    w = max(np.percentile(lat, 50) / 1000, 1e-6)               # p50 in seconds
    rps = 1 / w
    out(f"  {name}: W = p50 = {w * 1000:.0f} ms -> {rps:.1f} requests/s "
        f"-> {rps * 86400:,.0f} requests/day for one worker (upper bound)")
    return rps


def check_server(client, data, ctype):
    """Compare the running service with my notebook values."""
    h = client.get(f'{BASE}/health').json()
    out('Server settings vs my notebooks:')
    check('model loaded', h.get('model_loaded'))
    check(f'classes {EXPECTED_CLASSES} (Week 6)', h.get('classes') == EXPECTED_CLASSES,
          f"server says {h.get('classes')}")
    check('positive class is tumor', h.get('positive') == 'tumor')
    check(f'threshold {EXPECTED_THRESHOLD} (Week 6 operating point)',
          abs((h.get('threshold') or 0) - EXPECTED_THRESHOLD) < 1e-9,
          f"server says {h.get('threshold')}")
    check('resnet18, 224 px, grayscale',
          (h.get('backbone'), h.get('img_size'), h.get('grayscale')) == ('resnet18', 224, True))
    out(f"  spaCy entities enabled: {h.get('entities_enabled')} "
        '(optional: pip install spacy, python -m spacy download en_core_web_sm)')
    out()

    r = client.post(f'{BASE}/predict', files={'file': ('x.jpeg', data, ctype)}).json()
    p = r['probabilities']
    out(f"Your sample.jpeg: label={r['label']}  P(no_tumor)={p.get('no_tumor')}  "
        f"P(tumor)={p.get('tumor')}  flagged={r['positive_flag']} (threshold {r['threshold']})")
    out('  For comparison (Week 7 notebook): Te-gl_1.jpg gave P(tumor)=0.996, '
        'Te-no_1.jpg gave P(tumor)=0.017')
    out()

    rec = client.post(f'{BASE}/analyze', files={'file': ('x.jpeg', data, ctype)},
                      data={'note': NOTE}).json()
    blob = json.dumps(rec)
    out('Case record for the Week 7 note:')
    check('decision is REFER', rec.get('decision') == 'REFER', f"got {rec.get('decision')}")
    check('urgent terms headache, seizure, vomit',
          all(t in rec.get('urgent_terms', []) for t in ('headache', 'seizure', 'vomit')))
    check('masked: [CNIC] [PHONE] [MRN]',
          all(t in rec.get('note_masked', '') for t in ('[CNIC]', '[PHONE]', '[MRN]')))
    f = rec.get('fields', {})
    check('fields: 2.5 cm, age 46, T2, 10/11/2026',
          (f.get('tumour_size'), f.get('age'), f.get('sequence'), f.get('date')) ==
          (['2.5 cm'], ['46'], ['T2'], ['10/11/2026']))
    leaked = [x for x in LEAKS if x in blob]
    check('PII fragments found in the JSON: none', not leaked, f'leaked {leaked}' if leaked else '')
    out()


def bad_inputs(client, data, ctype):
    out('Bad-input checks (expected 415, 413, 400, 422, 422):')
    cases = [
        ('text file sent to /predict       ',
         lambda: client.post(f'{BASE}/predict',
                             files={'file': ('a.txt', b'hello', 'text/plain')}), 415),
        ('image of 5 MB + 1 byte           ',
         lambda: client.post(f'{BASE}/predict',
                             files={'file': ('b.jpg', io.BytesIO(b'0' * (MAX_BYTES + 1)),
                                             'image/jpeg')}), 413),
        ('bytes that are not an image      ',
         lambda: client.post(f'{BASE}/predict',
                             files={'file': ('c.jpg', b'not an image', 'image/jpeg')}), 400),
        ('no file at all                   ',
         lambda: client.post(f'{BASE}/predict'), 422),
        ('3,000-character note to /analyze ',
         lambda: client.post(f'{BASE}/analyze', files={'file': ('x.jpeg', data, ctype)},
                             data={'note': 'x' * 3000}), 422),
    ]
    for label, call, expected in cases:
        r = call()
        try:
            detail = str(r.json().get('detail'))[:60]
        except Exception:
            detail = ''
        ok = r.status_code == expected and 'Traceback' not in r.text
        check(f'{label}-> {r.status_code}', ok, detail)
    out()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    sample = find_sample()
    if sample is None:
        sys.exit(f'sample.jpeg not found in {HERE}\n'
                 'Copy one brain MRI image there and name it sample.jpeg '
                 '(check the name with: dir sample*).')
    data = sample.read_bytes()
    is_png = data[:8] == b'\x89PNG\r\n\x1a\n'
    ctype = 'image/png' if is_png else 'image/jpeg'

    ram = ram_gb()
    cpu, cores = cpu_model(), os.cpu_count()
    system = f'{platform.system()} {platform.release()}'
    out('=== Hardware ===')
    out(f'CPU : {cpu}  ({cores} logical cores)')
    out(f'RAM : {ram:.1f} GB' if ram else 'RAM : unknown (see Task Manager > Performance)')
    out(f'OS  : {system} | Python {platform.python_version()}')
    out('Server: 1 uvicorn process, 1 worker')
    out(f'Image : {sample.name} ({len(data) / 1024:.0f} KB)')
    if is_png and sample.suffix.lower() in ('.jpg', '.jpeg'):
        out('  Note: this file is really a PNG with a .jpeg name. The API still accepts it, '
            'but convert it to a true JPEG if you can.')
    out()

    with httpx.Client(timeout=60) as client:
        try:
            client.get(f'{BASE}/health').raise_for_status()
        except httpx.HTTPError as exc:
            sys.exit(f'Cannot reach {BASE} ({exc}).\n'
                     'Start the server first in another terminal: uvicorn app:app')

        out('=== Checks against my notebooks ===')
        check_server(client, data, ctype)

        out('=== Latency ===')
        lat_p = bench(client, '/predict', f'{BASE}/predict', data, ctype)
        lat_a = bench(client, '/analyze', f'{BASE}/analyze', data, ctype, with_note=True)
        out()

        out("=== Capacity (Little's law, L = lambda x W, L = 1 worker) ===")
        rps_p = littles_law(lat_p, '/predict')
        littles_law(lat_a, '/analyze')
        out()

        out('=== Bad inputs ===')
        bad_inputs(client, data, ctype)

    p50, p95 = np.percentile(lat_p, 50), np.percentile(lat_p, 95)
    out('=== Paste into the README (section 4) ===')
    out('| Hardware | CPU model | RAM | p50 | p95 | max | Throughput, 1 worker |')
    out('|---|---|---|---|---|---|---|')
    out(f'| {system}, {cores} logical cores | {cpu} | {ram:.1f} GB | {p50:.0f} ms | '
        f'{p95:.0f} ms | {lat_p.max():.0f} ms | {1000 / lat_p.mean():.1f} requests/s |'
        if ram else
        f'| {system}, {cores} logical cores | {cpu} | unknown | {p50:.0f} ms | '
        f'{p95:.0f} ms | {lat_p.max():.0f} ms | {1000 / lat_p.mean():.1f} requests/s |')
    out()
    out(f"Little's law: 1 / {p50 / 1000:.3f} s = {rps_p:.1f} requests/s "
        f'x 86,400 s = about {rps_p * 86400 / 1e6:.2f} million requests per day for one worker.')
    out('Bad input I tried: a text file sent to /predict. The API returned 415 '
        '"Upload a JPEG or PNG image".')
    failed = [n for n, ok in checks if not ok]
    out()
    out(f'{len(checks) - len(failed)} of {len(checks)} checks OK'
        + (f'. To look at: {"; ".join(failed)}' if failed else '. Everything matches my notebooks.'))

    (HERE / 'bench_results.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nSaved to bench_results.txt (paste it into the README).')


if __name__ == '__main__':
    main()
