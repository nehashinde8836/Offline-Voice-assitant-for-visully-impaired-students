content = open('frontend/index.html', encoding='utf-8').read()
checks = [
    ('DOCTYPE',           '<!DOCTYPE html>' in content),
    ('Noto font',         'Noto+Sans+Devanagari' in content),
    ('CSS vars',          '--send' in content),
    ('sidebar',           'class="sidebar"' in content),
    ('chat-header',       'class="chat-header"' in content),
    ('chat-box',          'id="chat-box"' in content),
    ('mic-btn',           'id="mic-btn"' in content),
    ('typing indicator',  'typing-dot' in content),
    ('score pill',        'score-pill' in content),
    ('quick chips',       'class="chip"' in content),
    ('text input',        'id="kb-input"' in content),
    ('sendText fn',       'async function sendText' in content),
    ('speakText fn',      'function speakText' in content),
    ('updateModeUI fn',   'function updateModeUI' in content),
    ('setupRecognition',  'function setupRecognition' in content),
    ('reset call',        '/reset' in content),
    ('closing html',      '</html>' in content),
]
all_ok = True
for name, ok in checks:
    print(f'  [{"OK" if ok else "FAIL"}] {name}')
    if not ok: all_ok = False
print(f'\nSize: {len(content)} chars, {content.count(chr(10))} lines')
print('All checks passed!' if all_ok else 'SOME FAILED')
