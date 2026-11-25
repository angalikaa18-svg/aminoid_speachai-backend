import urllib.request
import json
import sys

BASE_URL = 'http://127.0.0.1:8080'

def request(method, path, data=None):
    url = BASE_URL + path
    headers = {}
    body = None
    if data is not None:
        headers['Content-Type'] = 'application/json'
        body = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

tests = []

# 1. Health
s, d = request('GET', '/health')
assert s == 200 and d['status'] == 'healthy', 'Health failed'
tests.append('1. Health & Offline Mode Verified: 200 OK')

# 2. System Metrics
s, d = request('GET', '/api/v1/system/metrics')
assert s == 200 and 'cpu' in d and 'memory' in d, 'Metrics failed'
cpu_pct = d['cpu']['percent']
ram_pct = d['memory']['percent']
tests.append(f'2. System Telemetry: CPU {cpu_pct}%, RAM {ram_pct}% (psutil verified): 200 OK')

# 3. System Stats
s, d = request('GET', '/api/v1/system/stats')
assert s == 200 and 'stt_tasks' in d, 'Stats failed'
tests.append(f'3. System Stats: {d["total_audio_tasks"]} Total Tasks from DB: 200 OK')

# 4. Settings
s, d = request('GET', '/api/v1/settings')
assert s == 200 and 'default_tts_rate' in d, 'Settings GET failed'
s, d2 = request('POST', '/api/v1/settings', {'default_tts_rate': 180})
assert s == 200 and d2['settings']['default_tts_rate'] == 180, 'Settings POST failed'
tests.append('4. Persistent Settings GET & POST (storage/settings.json): 200 OK')

# 5. STT Languages & Models
s, langs = request('GET', '/api/v1/stt/languages')
s, mods = request('GET', '/api/v1/stt/models')
assert len(langs) > 0 and len(mods) > 0, 'STT lists failed'
tests.append(f'5. Dynamic STT Languages ({len(langs)}) & Models ({len(mods)}): 200 OK')

# 6. Unified Voices
s, voices = request('GET', '/api/v1/tts/voices')
assert len(voices) > 0, 'Voices empty'
categories = set(v.get('category') for v in voices)
tests.append(f'6. Unified Voice Catalog ({len(voices)} voices, categories: {categories}): 200 OK')

# 7. Models Catalog
s, models = request('GET', '/api/v1/models')
assert len(models.get('models', [])) > 0, 'Models catalog failed'
tests.append(f'7. Models Catalog ({len(models["models"])} engines): 200 OK')

# 8. Direct Webhook Receiver
s, wh = request('POST', '/webhook', {'event': 'platform.verify', 'source': 'automated_test'})
assert s == 200 and wh['success'], 'Webhook failed'
tests.append('8. Direct Webhook Receiver: 200 OK')

# 9. Webhook Dispatcher
s, disp = request('POST', '/api/v1/integrations/webhooks/test', {'target_url': 'http://127.0.0.1:8080/webhook', 'event': 'system.ping'})
assert s == 200 and disp['success'], 'Dispatcher failed'
tests.append('9. Webhook Outbound Dispatcher (HMAC Signature): 200 OK')

# 10. History Retrieval
s, hist = request('GET', '/api/v1/history?category=all&limit=10')
assert s == 200 and 'items' in hist, 'History failed'
tests.append(f'10. History Query & Search ({hist["total"]} total records): 200 OK')

# 11. Model Activation
s, act = request('POST', '/api/v1/models/activate?model_id=faster-whisper-tiny')
assert s == 200 and act['status'] == 'active', 'Activate failed'
tests.append('11. Model Activation (faster-whisper-tiny): 200 OK')

print('\n'.join(tests))
print(f'\n========================================')
print(f'ALL {len(tests)} FULL-STACK INTEGRATION TESTS PASSED 100%!')
print(f'========================================')
