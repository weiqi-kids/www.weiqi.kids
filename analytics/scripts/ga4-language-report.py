#!/usr/bin/env python3
"""
GA4 各語言流量報表（依網址路徑前綴判定語系，最準確）。

用法:
    python3 analytics/scripts/ga4-language-report.py [--days 28] [--top 15]

需求:
    - 服務帳號金鑰: /root/.config/ga4-insights/sa-key.json
      （SA: ga4-insights@yaocare.iam.gserviceaccount.com，已授權 weiqi.kids property）
    - Python 套件: jwt(PyJWT), requests（主機已安裝；無 google-auth）

說明:
    weiqi.kids 為 Docusaurus i18n 網站，各語系以路徑前綴區分
    （/en /ja /ko /es /pt /hi /id /ar /zh-cn /zh-hk；無前綴=繁中預設）。
    GA4 內建的 language 維度是「訪客瀏覽器語言」而非頁面語系，故不採用。
"""
import json, time, argparse
import jwt, requests

KEY = '/root/.config/ga4-insights/sa-key.json'
PROPERTY = '458470883'  # weiqi-kids（measurement ID G-16V1KSEH6W）
LOCALES = {
    'en': 'English', 'ja': '日本語', 'ko': '한국어', 'es': 'Español',
    'pt': 'Português', 'hi': 'हिन्दी', 'id': 'Bahasa Indonesia',
    'ar': 'العربية', 'zh-cn': '简体中文', 'zh-hk': '粵語(香港)',
}


def get_token():
    d = json.load(open(KEY))
    now = int(time.time())
    assertion = jwt.encode({
        'iss': d['client_email'],
        'scope': 'https://www.googleapis.com/auth/analytics.readonly',
        'aud': d['token_uri'], 'iat': now, 'exp': now + 3600,
    }, d['private_key'], algorithm='RS256')
    r = requests.post(d['token_uri'], data={
        'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        'assertion': assertion})
    r.raise_for_status()
    return r.json()['access_token']


def run_report(token, body):
    url = f'https://analyticsdata.googleapis.com/v1beta/properties/{PROPERTY}:runReport'
    r = requests.post(url, headers={'Authorization': f'Bearer {token}',
                                    'Content-Type': 'application/json'},
                      data=json.dumps(body))
    js = r.json()
    if 'error' in js:
        raise SystemExit('GA4 API error: ' + json.dumps(js['error'], ensure_ascii=False))
    return js


def locale_of(path):
    seg = path.strip('/').split('/')[0] if path.strip('/') else ''
    return LOCALES.get(seg, '繁體中文(預設)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=28, help='往前幾天（預設 28）')
    ap.add_argument('--top', type=int, default=15, help='熱門頁顯示筆數（預設 15）')
    args = ap.parse_args()
    tok = get_token()
    rng = {'startDate': f'{args.days}daysAgo', 'endDate': 'today'}

    js = run_report(tok, {
        'dateRanges': [rng],
        'dimensions': [{'name': 'pagePath'}],
        'metrics': [{'name': 'screenPageViews'}, {'name': 'sessions'}, {'name': 'activeUsers'}],
        'limit': 5000,
    })
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    pages = []
    for row in js.get('rows', []):
        path = row['dimensionValues'][0]['value']
        pv, ss, us = (int(row['metricValues'][i]['value']) for i in range(3))
        a = agg[locale_of(path)]
        a[0] += pv; a[1] += ss; a[2] += us
        pages.append((path, pv))
    total = [sum(v[i] for v in agg.values()) for i in range(3)]

    print(f'\n=== weiqi.kids 各語言流量（近 {args.days} 天，依路徑前綴）===')
    print(f"{'語言':<18}{'瀏覽':>7}{'工作階段':>10}{'使用者':>9}{'瀏覽佔比':>10}")
    for k, v in sorted(agg.items(), key=lambda x: -x[1][0]):
        pct = v[0] / total[0] * 100 if total[0] else 0
        print(f"{k:<18}{v[0]:>7}{v[1]:>10}{v[2]:>9}{pct:>9.1f}%")
    print(f"{'合計':<18}{total[0]:>7}{total[1]:>10}{total[2]:>9}")

    print(f'\n=== 熱門頁 Top {args.top} ===')
    for path, pv in sorted(pages, key=lambda x: -x[1])[:args.top]:
        print(f'{pv:>5}  {path}')


if __name__ == '__main__':
    main()
