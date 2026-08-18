items: 105

## temperature 0.7  (n=105)
overall: org_score 0.593 | our_score 0.638 | agree(>=.5) 0.876 | corr 0.757
outlen : org median 236 | ours 136 | median ratio 0.55 | corr(log) 0.194
inlen  : org median 140 | ours 107 | median diff(org-ours) +9

| family | n | org_score | our_score | agree | corr | org_out(med) | our_out(med) | ratio(med) | ratio IQR | in diff(med) |
|---|---|---|---|---|---|---|---|---|---|---|
| aime | 15 | 0.98 | 0.97 | 1.00 | -0.07 | 270 | 126 | 0.47 | 0.45-0.57 | +3 |
| belebele | 15 | 0.87 | 0.83 | 0.93 | 0.77 | 18 | 35 | 2.27 | 1.25-3.64 | +9 |
| code | 15 | 0.20 | 0.40 | 0.47 | -0.08 | 28 | 222 | 7.62 | 5.42-8.94 | +219 |
| gsm8k_or_other | 15 | 0.73 | 0.73 | 1.00 | 1.00 | 294 | 144 | 0.49 | 0.45-0.52 | +3 |
| hrmcr | 15 | 0.03 | 0.10 | 0.93 | 0.89 | 264 | 138 | 0.53 | 0.48-0.60 | +19 |
| ruletaker | 15 | 0.80 | 0.83 | 0.93 | 0.47 | 260 | 180 | 0.81 | 0.55-10.43 | +35 |
| truthfulqa | 15 | 0.53 | 0.60 | 0.87 | 0.79 | 256 | 85 | 0.33 | 0.29-0.42 | +7 |

disagreements (org>=.5 vs ours):
  dev-0505     belebele       org=0.00 ours=0.50 gold='B' preds=['B', 'C']
  dev-0565     code           org=0.00 ours=1.00 gold='[]' preds=['[]', '[]']
  train-1214   code           org=0.00 ours=1.00 gold='[3, 5, 4, 3, 2, 1, 0' preds=['[3, 5, 4, 3, 2, 1, 0', '[3, 5, 4, 3, 2, 1, 0']
  train-0792   code           org=0.50 ours=0.00 gold='[5, 4, 3, 2, 1, 0], ' preds=['[3, 5, 4, 3, 2, 1, 0', '[3, 5, 4, 3, 1, 0]']
  train-0165   code           org=1.00 ours=0.00 gold="'zjegiymjc', 'j', 2" preds=['', 'abcd']
  dev-0297     code           org=0.00 ours=1.00 gold='[-3, 1, 7, -1]' preds=['[-3, 1, 7, -1]', '[-3, 1, 7, -1]']
  train-1225   code           org=0.00 ours=1.00 gold="'zzzzzabc'" preds=['zzzzzabc', 'zzzzzabc']
  dev-0270     code           org=0.00 ours=1.00 gold='1' preds=['1', '1']
  dev-0339     code           org=0.50 ours=0.00 gold="{}, 'hbd'" preds=["{'h': 'h', 'e': 'b',", "{'h': 'h', 'e': 'b',"]
  dev-0759     hrmcr          org=0.00 ours=0.50 gold="['양', '소']" preds=['개띠', '소띠입니다']
  train-0191   ruletaker      org=0.00 ours=1.00 gold='True' preds=['True', 'True']
  train-1715   truthfulqa     org=0.50 ours=0.00 gold='B' preds=['A', 'A']
  dev-0853     truthfulqa     org=0.00 ours=1.00 gold='B' preds=['B', 'B']