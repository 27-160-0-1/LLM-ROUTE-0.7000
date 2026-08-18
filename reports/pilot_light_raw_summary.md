items: 80

## temperature 0.7  (n=80)
overall: org_score 0.584 | our_score 0.481 | agree(>=.5) 0.800 | corr 0.664
outlen : org median 225 | ours 130 | median ratio 0.67 | corr(log) 0.111
inlen  : org median 161 | ours 89 | median diff(org-ours) +39

| family | n | org_score | our_score | agree | corr | org_out(med) | our_out(med) | ratio(med) | ratio IQR | in diff(med) |
|---|---|---|---|---|---|---|---|---|---|---|
| aime | 15 | 0.98 | 0.87 | 0.93 | -0.12 | 270 | 66 | 0.22 | 0.19-0.35 | +30 |
| belebele | 15 | 0.87 | 0.67 | 0.87 | 0.19 | 18 | 67 | 4.10 | 2.92-5.16 | +48 |
| code | 15 | 0.20 | 0.23 | 0.53 | 0.16 | 28 | 244 | 6.64 | 4.56-8.77 | +265 |
| gsm8k_or_other | 15 | 0.73 | 0.70 | 0.93 | 0.87 | 294 | 126 | 0.34 | 0.29-0.63 | +30 |
| hrmcr | 15 | 0.03 | 0.03 | 0.87 | -0.07 | 264 | 140 | 0.56 | 0.49-0.60 | +39 |
| ruletaker | 5 | 0.90 | 0.20 | 0.40 | 0.41 | 307 | 252 | 0.67 | 0.61-0.82 | +72 |

disagreements (org>=.5 vs ours):
  train-0253   aime           org=1.00 ours=0.00 gold='9' preds=['7', '5']
  train-0209   belebele       org=1.00 ours=0.00 gold='C' preds=['따라서, 현재 올림픽 정식 종목으로 ', '따라서, 올림픽 종목에서 제외된 스포']
  train-1128   belebele       org=0.00 ours=1.00 gold='C' preds=['변은 제공된 선택지 중에서는 명확하게', '변을 찾는다면, **"드룩갈종의 화재']
  dev-0565     code           org=0.00 ours=1.00 gold='[]' preds=['But for the placehol', '']
  train-0792   code           org=0.50 ours=0.00 gold='[5, 4, 3, 2, 1, 0], ' preds=['', 'But if you want the ']
  train-0165   code           org=1.00 ours=0.00 gold="'zjegiymjc', 'j', 2" preds=['Note**: The placehol', '']
  dev-0297     code           org=0.00 ours=0.50 gold='[-3, 1, 7, -1]' preds=[':', 'Therefore, `f([1, 7,']
  train-0459   code           org=0.00 ours=0.50 gold='[2, 3, 4, 6, -2]' preds=['', 'If you have a differ']
  dev-0270     code           org=0.00 ours=0.50 gold='1' preds=['Therefore, the expec', 'So, it seems like th']
  dev-0339     code           org=0.50 ours=0.00 gold="{}, 'hbd'" preds=['This will pass the a', 'This example demonst']
  train-1483   gsm8k_or_other org=0.00 ours=0.50 gold='1800' preds=['1260', '1800']
  dev-0104     hrmcr          org=0.00 ours=0.50 gold='2006.8.16' preds=['입니다', '즉, 생일로부터 2일 후의 음력 날짜']
  train-1532   hrmcr          org=0.50 ours=0.00 gold="['말']" preds=[':', '- **B의 띠**: **원숭이띠']
  train-1649   ruletaker      org=0.50 ours=0.00 gold='False' preds=['Bob is green, not ro', '']
  train-0445   ruletaker      org=1.00 ours=0.00 gold='True' preds=['Yes', 'Harry is not red']
  train-1562   ruletaker      org=1.00 ours=0.00 gold='False' preds=['', 'The final answer is ']