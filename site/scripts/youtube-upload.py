#!/usr/bin/env python3
"""上傳影片到協會 YouTube 頻道（OAuth 裝置授權流程＋可續傳上傳）。

憑證放在 ~/.config/weiqi-kids/youtube-client.json（{"client_id","client_secret"}），
授權後的 refresh token 存在 ~/.config/weiqi-kids/youtube-token.json，都不進 git。

用法：
  youtube-upload.py auth                 # 取得驗證碼，使用者到 google.com/device 授權
  youtube-upload.py upload plan.json     # 依清單上傳；已上傳的會記在 plan.json 的 videoId
"""
import json, os, sys, time, urllib.request, urllib.parse

CFG = os.path.expanduser('~/.config/weiqi-kids')
CLIENT = os.path.join(CFG, 'youtube-client.json')
TOKEN = os.path.join(CFG, 'youtube-token.json')
SCOPE = 'https://www.googleapis.com/auth/youtube'


def post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), method='POST')
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


def auth():
    c = json.load(open(CLIENT))
    d = post('https://oauth2.googleapis.com/device/code', {'client_id': c['client_id'], 'scope': SCOPE})
    if 'user_code' not in d:
        sys.exit(f'無法取得驗證碼：{d}')
    print(f"請到 {d['verification_url']} 輸入驗證碼：{d['user_code']}", flush=True)
    deadline = time.time() + d['expires_in']
    while time.time() < deadline:
        time.sleep(d.get('interval', 5))
        t = post('https://oauth2.googleapis.com/token', {
            'client_id': c['client_id'], 'client_secret': c['client_secret'],
            'device_code': d['device_code'], 'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'})
        if 'refresh_token' in t:
            json.dump(t, open(TOKEN, 'w')); os.chmod(TOKEN, 0o600)
            print('授權完成', flush=True); return
        if t.get('error') not in ('authorization_pending', 'slow_down'):
            sys.exit(f'授權失敗：{t}')
    sys.exit('驗證碼已過期')


def access_token():
    c, t = json.load(open(CLIENT)), json.load(open(TOKEN))
    r = post('https://oauth2.googleapis.com/token', {
        'client_id': c['client_id'], 'client_secret': c['client_secret'],
        'refresh_token': t['refresh_token'], 'grant_type': 'refresh_token'})
    return r['access_token']


def api(method, url, token, body=None, headers=None, raw=False):
    """呼叫 API；raw=True 時回傳 (headers, body) 供可續傳上傳取 Location。"""
    h = {'Authorization': f'Bearer {token}', **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if body is not None: h['Content-Type'] = 'application/json; charset=UTF-8'
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    with urllib.request.urlopen(req) as r:
        payload = r.read()
        return (dict(r.headers), payload) if raw else (json.loads(payload) if payload else {})


def upload_one(token, item):
    meta = {
        'snippet': {'title': item['title'], 'description': item['description'], 'tags': item.get('tags', []),
                    'categoryId': item.get('categoryId', '27'), 'defaultLanguage': 'zh-TW', 'defaultAudioLanguage': 'zh-TW'},
        'status': {'privacyStatus': item.get('privacy', 'public'), 'selfDeclaredMadeForKids': False, 'embeddable': True},
    }
    size = os.path.getsize(item['file'])
    headers, _ = api('POST', 'https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status',
                     token, meta, {'X-Upload-Content-Length': str(size), 'X-Upload-Content-Type': 'video/*'}, raw=True)
    session = headers['Location']
    with open(item['file'], 'rb') as f:
        req = urllib.request.Request(session, data=f, method='PUT',
                                     headers={'Authorization': f'Bearer {token}', 'Content-Length': str(size), 'Content-Type': 'video/*'})
        with urllib.request.urlopen(req, timeout=3600) as resp:
            return json.load(resp)['id']


def add_to_playlist(token, playlist_id, video_id):
    api('POST', 'https://www.googleapis.com/youtube/v3/playlistItems?part=snippet', token,
        {'snippet': {'playlistId': playlist_id, 'resourceId': {'kind': 'youtube#video', 'videoId': video_id}}})


def ensure_playlist(token, plan):
    for pl in plan.get('playlists', []):
        if pl.get('id'): continue
        r = api('POST', 'https://www.googleapis.com/youtube/v3/playlists?part=snippet,status', token,
                {'snippet': {'title': pl['title'], 'description': pl.get('description', ''), 'defaultLanguage': 'zh-TW'},
                 'status': {'privacyStatus': 'public'}})
        pl['id'] = r['id']


def upload(plan_path):
    plan = json.load(open(plan_path))
    token = access_token()
    ensure_playlist(token, plan); json.dump(plan, open(plan_path, 'w'), ensure_ascii=False, indent=1)
    playlists = {p['key']: p['id'] for p in plan.get('playlists', [])}
    for item in plan['videos']:
        if item.get('videoId'): continue
        print(f"上傳：{item['title']}", flush=True)
        token = access_token()
        item['videoId'] = upload_one(token, item)
        json.dump(plan, open(plan_path, 'w'), ensure_ascii=False, indent=1)
        pid = playlists.get(item.get('playlist'))
        if pid: add_to_playlist(token, pid, item['videoId'])
        print(f"  完成 https://youtu.be/{item['videoId']}", flush=True)


if __name__ == '__main__':
    {'auth': auth, 'upload': lambda: upload(sys.argv[2])}[sys.argv[1]]()
