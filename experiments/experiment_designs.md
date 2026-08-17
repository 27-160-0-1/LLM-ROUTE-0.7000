<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# 신규 실험 설계 8종 (E26 ~ E33)

작성일 2026-08-16. 기준선: 가중 CV-EV **0.6976** (gain α=0.5, kNN k=16),
tier별 안전계수 0.98 / 0.89 / 0.88, fast oracle 격차 **+0.09**.

## 0. 설계 전 확인한 구조적 사실 (설계 근거)

harness(`colab-sweep/bundle/colab_sweep.py`)를 읽고 확인한, 아래 설계들이
공통으로 딛고 있는 사실 4가지:

1. **메타 GBM의 입력은 58차원뿐이다.** `dense(30) + family_onehot(9) +
   legacy(6) + ridge_oof(6) + knn(7)`. 즉 16,414차원 해시 어휘 정보는
   ridge의 스칼라 6개와 kNN의 스칼라 7개로 **압축된 뒤에야** 트리에 닿는다.
   트리는 "특정 어휘 × 길이 × family" 상호작용을 표현할 통로가 아예 없다.
   → E28의 근거. (E19가 기각된 것은 *ridge의 해시 해상도*를 키운 것이지
   *트리의 입력*을 넓힌 것이 아니다. 축이 다르다.)
2. **fast tier(multiplier 1.25)에서 할당이 소비하는 것은 순위뿐이다.**
   `allocate()`는 `ps - λ·pc`의 argmax이고, 2지 선택 구간에서는 정확히
   효율 `Δs/Δc` 기준 순위와 동치다. 예산이 빠듯할수록 **상위 10~20%
   꼬리의 순위 정밀도**만 점수에 반영되고, 나머지 80%의 회귀 오차는
   점수에 아무 영향이 없다. 현재 gain 헤드는 제곱손실 회귀라 전 구간
   평균오차를 줄이도록 학습된다 — 목적 불일치.
   → E26 · E27 · E30의 근거.
3. **δ 타깃은 영-과잉(zero-inflated)이다.** score∈{0,.25,.5,.75,1}이므로
   δ₁=s₁−s₀ 는 0.25 간격 9값이고, 두 모델이 같이 맞거나 같이 틀린 문항이
   0에 몰린다. 제곱손실 단일 회귀는 조건부 평균으로 수축해 꼬리를 뭉갠다.
   → E26의 근거. (E21 순서형은 *모델별 절대 점수*의 누적임계였고,
   여기서 다루는 것은 *모델 간 차이*의 부호/크기 분해다. 타깃이 다르다.)
4. **EV 손실의 절반은 안전계수에서 나온다.** E09에서 확립됐듯 안전계수는
   비용 예측 오차의 분산 때문에 0.88~0.98로 눌려 있다. 비용 예측의
   *체계적 편향*(Jensen)과 *예측 가능한 이분산*을 각각 제거하면 같은
   초과확률에서 안전계수를 올릴 수 있고, 이는 점수에 직결된다.
   → E31 · E32의 근거. balanced/premium(안전계수 0.89/0.88)에서 여지가 크다.

### 비용 참조표 (Colab CPU 기준, E20 실측 fold당 ~7분에서 환산)

| 재계산 범위 | 1 config 소요 | 해당 실험 |
|---|---:|---|
| 전체 파이프라인 (kNN행 재계산 + 8헤드 재학습) | ~35분 | E29 |
| 메타 단계만 (`fold_cache` 재사용, 헤드만 재학습) | ~8분 | E26·E27·E28·E31·E32·E33 |
| 사후처리만 (CV 예측 캐시 위에서 그리드) | 수십 초 | E30, 각 실험의 α/κ/τ 그리드 |

> **권장 실행 순서**: E30 → E31 → E26 → E27 → E28 → E32 → E33 → E29.
> 앞의 둘은 사후처리·메타전용이라 하루 안에 판정이 나고, 뒤로 갈수록
> 재계산 비용이 커진다. E30/E31이 안전계수를 올려주면 그 위에서
> fast 계열(E26~E28)의 이득이 더 크게 측정된다.

---

## E26. 부호 분해 이득 헤드 (help/hurt 분류 × 조건부 크기)

- **실행 위치**: Colab CPU (메타 전용, `fold_cache` 재사용)
- **축**: 타깃 설계 — **fast 격차 직접 겨냥** ★

### 가설

E14의 gain 헤드는 영-과잉·좌우대칭 타깃에 제곱손실 단일 회귀를 걸어서
예측이 조건부 평균(≈0)으로 수축하고, 정작 할당이 쓰는 **상위 꼬리의 순위**가
뭉개진다. 이득을 `P(도움) · E[크기|도움] − P(손해) · E[크기|손해]`로 분해하면
각 성분이 자기 부분모집단에서만 학습되므로 꼬리가 살아남는다. E21(모델별
절대 점수의 누적임계)과는 타깃 자체가 다르다 — 여기서는 *모델 간 차이*의
부호를 모델링한다.

### 방법

```python
# 0) 사전 확인: δ 분포의 0 질량 비율을 먼저 출력 (40~60% 예상). 25% 미만이면 중단.
for g, (a, b) in enumerate([(0, 1), (1, 2)]):            # light→3.1, 3.1→K1
    d = true_s[:, b] - true_s[:, a]                       # ∈ {-1,...,1}, 0.25 간격
    # fold-train 안에서만 적합 (X_meta_fit / X_meta_hold 규약 그대로)
    p_pos = HistGBClassifier(**CFG).fit(Xf, (d[f] > 0))   # 도움 확률
    p_neg = HistGBClassifier(**CFG).fit(Xf, (d[f] < 0))   # 손해 확률
    m_pos = HistGBRegressor(**CFG_S).fit(Xf[d[f] > 0],  d[f][d[f] > 0])
    m_neg = HistGBRegressor(**CFG_S).fit(Xf[d[f] < 0], -d[f][d[f] < 0])
    dhat[:, g] = (p_pos.predict_proba(Xh)[:, 1] * m_pos.predict(Xh)
                  - p_neg.predict_proba(Xh)[:, 1] * m_neg.predict(Xh))
# 1) 기존 recon 자리에 dhat을 넣고, 기존 δ회귀 헤드와 β로 혼합
dhat_mix = (1 - BETA) * delta_head_pred + BETA * dhat      # BETA ∈ {0,.25,.5,.75,1}
recon = column_stack([m0, m0 + dhat_mix[:,0], m0 + dhat_mix[:,0] + dhat_mix[:,1]])
meta_all[:, :3] = (1 - GAIN_ALPHA) * meta_all[:, :3] + GAIN_ALPHA * recon
```

- `CFG_S`는 부분모집단(n≈600~1,200)용으로 축소: `max_leaf_nodes=7,
  max_iter≤80, min_samples_leaf=20`, early stopping 유지.
- **export 가능성**: 분류 헤드도 트리 배열은 동일하게 export되고, 런타임은
  raw score에 `1/(1+math.exp(-x))`만 적용하면 된다 (stdlib). 조건부 크기
  헤드는 변환 불필요. `similarity.evaluate_trees` 그대로 사용.
- **판정 기준**: β 곡선이 **단봉·매끄러움**일 것 (E14의 α 곡선이 신뢰를 준
  방식과 동일). E21처럼 양 끝점보다 중간이 낮으면 노이즈로 보고 기각.

### 소요·기대

델타 헤드가 2개 → 8개로 늘어 메타 적합 시간 약 2배(~15분/config).
β 5점 그리드는 사후처리라 추가 비용 미미. **총 약 1.5시간.**
기대 이득 **+0.000 ~ +0.004 EV**, 대부분 fast에 집중. 아티팩트는 델타 트리
블록이 4배가 되므로 heavy 블록 크기·QEMU 시간(현재 46~61초) 재확인 필요.

---

## E27. 랭크 변환 효율 헤드 (할당 목적함수와 정렬된 학습 타깃)

- **실행 위치**: Colab CPU (메타 전용)
- **축**: 메타모델 구조 (트리 계열 내) — **fast 격차 직접 겨냥** ★

### 가설

할당이 실제로 소비하는 양은 점수도 이득도 아닌 **효율 `Δs/Δc`의 순위**다.
순위 지표를 제곱손실 회귀로 우회 학습하면 두꺼운 꼬리를 가진 `1/Δc`가
손실을 지배해 정작 순위가 망가진다. 타깃을 fold-train 내 **경험적 백분위
순위**로 변환하면 분포 무관·이상치 무관이 되고 제곱손실이 곧 순위 손실의
매끄러운 대리가 된다 — 트리 구조·export 경로는 전혀 바뀌지 않는다.
(LambdaMART류 pairwise 손실은 sklearn HistGBM에 없고 export도 복잡해지므로,
동일 효과를 타깃 변환만으로 얻는 설계다.)

### 방법

```python
for g, (a, b) in enumerate([(0, 1), (1, 2)]):
    ds = true_s[:, b] - true_s[:, a]
    dc = np.maximum(true_c[:, b] - true_c[:, a], DC_FLOOR)   # DC_FLOOR = 5% 분위수
    eff = ds / dc
    # fold-train 안에서만 순위화 → [0,1] 타깃, 동시에 역변환용 분위함수 저장
    r = rankdata(eff[f], method="average") / (len(f) - 1)
    head = HistGBRegressor(**CFG).fit(Xf, r)
    q = np.quantile(eff[f], np.linspace(0, 1, 65))            # 65-노드 계단 LUT
    eff_hat = np.interp(np.clip(head.predict(Xh), 0, 1), np.linspace(0,1,65), q)
    dhat[:, g] = eff_hat * (pc[:, b] - pc[:, a])              # 예측 비용차로 되곱함
recon = column_stack([m0, m0 + dhat[:,0], m0 + dhat[:,0] + dhat[:,1]])
meta_all[:, :3] = (1 - A) * meta_all[:, :3] + A * recon       # A ∈ {0,.3,.5,.7}
```

- 변형 1: `sample_weight`를 fast 예산 절단점 근방(효율 백분위 0.70~0.95)에서
  2배로 줘 **결정 경계 집중 학습**. 변형 2: fast 전용 헤드를 따로 두고
  tier별로 다른 recon 사용(현 구조가 이미 tier별 blend를 가지므로 자연스럽다).
- **export 가능성**: 트리 배열 + 65개 부동소수 LUT + `Δĉ` 곱 — 전부 stdlib.
  선형보간은 이분탐색 한 번.

### 소요·기대

메타 전용 ~8분/config × (기본 + 가중변형 + tier전용) × A 4점 ≈ **1.5시간**.
기대 이득 **+0.001 ~ +0.005 EV**. fast에 거의 전량 반영될 것으로 보며,
premium(multiplier 4.0, 예산 여유)에서는 이득이 없거나 소폭 손해일 수 있으니
tier별 채택을 허용할 것.

---

## E28. 메타 트리용 지도 선택 어휘 사전 특징 (허용된 특징 실험 1건)

- **실행 위치**: Colab CPU (메타 전용) + 선택 단계는 로컬 GPU(CuPy)로 가속 가능
- **축**: 특징 (1건만 사용) — **fast 격차 직접 겨냥** ★

### 가설

E19가 기각한 것은 *ridge의 해시 해상도*(bins·stride)였다. 여기서 넓히는 것은
**메타 트리의 입력 통로**다 (§0-1). 현재 트리는 어휘를 ridge/kNN 스칼라로만
보므로 "이 키워드가 있으면서 길이가 길 때만 light가 실패" 같은 상호작용을
원리적으로 표현할 수 없다. δ₁에 대해 지도 선택한 소수(≈48개)의 명시적
존재-지시 특징을 주면 그 통로가 열린다. 해시 공간 확대(노이즈 추가)와
반대로, **선택은 차원을 늘리지 않고 신호만 통과**시킨다.

### 방법

```python
# 각 outer fold의 fold-train만으로 선택 (누출 차단이 이 실험의 핵심 리스크)
d1 = true_s[f, 1] - true_s[f, 0]
for tok in vocab(min_df=30):                       # 기존 토크나이저의 w1/w2 그대로
    m = presence[f, tok]
    t[tok] = (d1[m].mean() - d1[~m].mean()) / se(d1, m)     # welch t 통계량
sel_gain  = top_k(|t|, 24)
sel_score = top_k(|t_s0|, 24)                      # s0(light 절대점수)에 대해서도 24개
# 안정성 필터: fold-train을 5분할해 4/5 이상에서 선택된 토큰만 채택
sel = [tok for tok in sel_gain + sel_score if inner_selection_count[tok] >= 4]
X_meta_fit  = hstack([X_meta_fit,  presence[f][:, sel].astype(float)])   # 58 → ~106
X_meta_hold = hstack([X_meta_hold, presence[h][:, sel].astype(float)])
```

- K ∈ {16, 32, 48, 96} 스윕. min_df ∈ {30, 60}.
- **export 가능성**: 선택된 토큰의 FNV-1a 해시 목록만 아티팩트에 저장(수 KB).
  런타임은 이미 같은 정규식으로 토큰을 뽑고 있으므로, 토큰 집합에 대한
  set 교집합 한 번 — **추가 비용 사실상 0**.
- **위험**: 선택 불안정성과 fold 누출. 안정성 필터와 fold-pure 선택 두 가지를
  반드시 지킬 것. 선택된 토큰 목록을 fold별로 덤프해 육안 검토하면 신호가
  진짜인지(출처 템플릿 문구인지 난이도 신호인지) 즉시 판별된다 — 이
  부산물만으로도 실험 가치가 있다.

### 소요·기대

선택 단계는 (어휘 3만 × 2,640) 불리언 행렬 통계라 로컬 GPU에서 초 단위,
CPU로도 fold당 1~2분. 메타 재적합 포함 ~12분/config × 8 = **약 1.5시간**.
기대 이득 **+0.001 ~ +0.005 EV**. 실패해도 "어떤 어휘가 light 실패를
예측하는가"라는 해석 산출물이 남아 이후 family 규칙 개선에 재사용된다.

---

## E29. 이득-판별 지도 메트릭 kNN (표현은 그대로, 거리만 지도 학습)

- **실행 위치**: 로컬 GPU (RTX 2050 / CuPy) 우선, 확인 런만 Colab CPU
- **축**: kNN · 유사도 구조

### 가설

현재 kNN 거리는 비지도 idf 가중 코사인이라 **"같은 출처/주제"** 를 잘 찾는다.
그러나 우리가 필요한 이웃은 **"light가 같은 이유로 실패하는 문항"** 이다.
두 개념은 다르다. 표현(해시 문자 3·4·5-gram)은 그대로 두고 **bin별 가중치만
δ₁에 대해 지도 학습**하면, E15(word-kNN: 표현 추가)나 E25(임베딩 teacher:
표현 교체)와 완전히 다른 메커니즘이며, 둘 다 실패한 이유(표현은 이미 충분히
좋다 — E25가 이를 강하게 시사)와 오히려 정합한다.

### 방법

```python
# fold-train만으로 bin 가중치 학습 (32,768 bins)
r = corr(tfidf_train[:, j], d1_train) for each j        # CuPy: 한 번의 sparse matmul
w = 1.0 + GAMMA * np.abs(r) / np.abs(r).std()           # GAMMA ∈ {0,.5,1,2,4}
idf_eff = idf * w                                        # 기존 idf 자리에 곱해 넣음
# 이후는 기존 knn_query 경로 그대로 (재정규화 → top-256 성분 → 코사인 top-16)
knn_rows = knn_query_with(idf_eff, exclude=self)
```

- 변형 A(단일 메트릭): 위 그대로, 기존 kNN 7열을 대체.
- 변형 B(이중 메트릭): 기존 비지도 메트릭 kNN 7열은 **유지**하고, gain 메트릭
  kNN을 별도 조회해 `[δ₁,δ₂ 이웃평균, top1 유사도]` 3열을 **추가**(58→61).
  두 메트릭이 서로 다른 이웃을 보므로 상보적일 가능성이 높다.
  **변형 B를 본안으로 본다.**
- 대안 가중: 상관 대신 δ₁에 대한 ridge 계수 절댓값(이미 GPU 학습 코드 존재).
- **export 가능성**: 아티팩트에 이미 bin별 idf 벡터가 있다 — 숫자만 바뀌거나
  두 번째 벡터가 추가될 뿐, **런타임 코드 변경 0, 추가 연산 0(변형 A)**.
  변형 B는 kNN 조회 1회 추가이므로 heavy 블록·QEMU 시간 재측정 필요.

### 소요·기대

kNN 행 전체 재계산이 필요해 전체 파이프라인 재실행: Colab 기준 ~35분/config.
GAMMA 5점 × 변형 2 = 10 config → **Colab 6시간** 또는 **로컬 CuPy 약 1시간**
(2,640 × 32,768 희소는 RTX 2050 메모리에 충분히 들어감; 로컬에서 후보를
2~3개로 좁힌 뒤 Colab에서 확인 런 권장). 기대 이득 **+0.000 ~ +0.004 EV**.
GAMMA=0이 기준선과 정확히 일치하는지 먼저 확인해 하니스 정합성을 검증할 것.

---

## E30. 이득의 등장성(isotonic) 보정 — 최적화자의 저주 제거

- **실행 위치**: 로컬 CPU (기존 CV 예측 캐시 위 사후처리)
- **축**: 보정 · 할당

### 가설

할당은 이득이 **과대예측된 문항을 골라내는 선택 편향**을 갖는다(optimizer's
curse). 따라서 선택된 집합의 실현 이득은 예측 이득보다 체계적으로 낮고, 그
편향의 크기는 예측 이득의 수준에 따라 다르다. `E[δ_true | δ̂]`를 fold-pure
등장성 회귀로 추정해 대입하면 이 편향이 제거된다. 등장성은 단조라 이득 단독
순위는 보존하지만, 할당은 **`δ̂/Δĉ` 로 순위**를 매기므로 비선형 단조 변환이
실제 선택 집합을 바꾼다 — 이것이 작동 메커니즘이다. E09(전역 스칼라 안전계수)
와 E08(문항별 비용 상한, 기각)과는 건드리는 대상이 다르다.

### 방법

```python
# CV 예측(meta_all)과 실제(true_s)만 있으면 됨 — 재학습 없음
for g in (0, 1):
    dhat, dtrue = pred_delta[:, g], true_delta[:, g]
    for k in range(5):                                  # fold-pure: 다른 fold로 적합
        iso = IsotonicRegression(out_of_bounds="clip").fit(dhat[~fold_k], dtrue[~fold_k])
        cal[fold_k, g] = iso.predict(dhat[fold_k])
# 변형 2D: 예측 비용차 10분위별로 별도 등장성 (이득×비용 상호작용 포착)
for q in range(10):
    iso_q = IsotonicRegression().fit(dhat[in_q & ~fold_k], dtrue[in_q & ~fold_k])
recon = column_stack([m0, m0 + cal[:,0], m0 + cal[:,0] + cal[:,1]])
meta_all[:, :3] = (1-A)*meta_all[:, :3] + A*recon ; 안전계수 그리드 재스윕
```

- 등장성 대신 **단조 PAV 후 64노드로 축약**해 export (계단 LUT).
- 진단 산출물: `δ̂` 10분위별 `mean(δ_true) - mean(δ̂)` 표. 이 표가 평평하면
  편향이 없다는 뜻이므로 즉시 기각하고 다음 실험으로 넘어갈 수 있다 —
  **비용 대비 정보량이 가장 높은 실험**이라 1순위로 권장한다.
- **export 가능성**: (threshold, value) 쌍 ~64개 × 2 헤드. 자명하게 stdlib.

### 소요·기대

재학습 없음. **전체 30분 이내**(대부분 안전계수 재스윕). 기대 이득
**+0.000 ~ +0.003 EV**. 이득이 0이더라도 §0-2 가설(꼬리 편향)의 참/거짓을
확정해 E26·E27의 사전확률을 갱신해 주므로, 결과와 무관하게 먼저 돌릴 것.

---

## E31. 비용 타깃을 비율로 전환 + Jensen/smearing 보정

- **실행 위치**: Colab CPU (메타 전용)
- **축**: 비용 예측 개선

### 가설

두 가지 체계적 결함이 동시에 존재한다. (a) 비용 헤드 3개가 `log c_m`을
**독립적으로** 회귀하는데, 오차의 지배적 성분("이 문항의 답이 얼마나 길까")은
세 모델에 **공통**이다. 예산은 `Σc_pick / Σc_light` 비율로만 판정되므로 공통
성분은 원래 상쇄되어야 하는데, 독립 회귀는 이를 상쇄시키지 못하고 분산으로
남긴다. (b) `exp(μ̂)`는 조건부 **기하평균**(≈중앙값)이라 `E[c]`를 `exp(σ²/2)`
배만큼 **체계적으로 과소추정**한다. 예산 회계는 산술합이므로 이 편향이 곧
초과확률이고, 곧 안전계수 억압이다.

### 방법

```python
# (a) 비율 타깃
y_cost = column_stack([log(c0), log(c1/c0), log(c2/c1)])   # 절대 1개 + 비율 2개
heads  = [HistGBRegressor(**CFG).fit(Xf, y_cost[f, j]) for j in range(3)]
lc = heads[0].predict(Xh)
pc_log = column_stack([lc, lc + heads[1].predict(Xh), lc + heads[1](..) + heads[2](..)])
# → 비용 단조성(light ≤ 3.1 ≤ K1)이 구조적으로 보장됨 (현재는 사후 강제 중)

# (b) Duan smearing: fold-train 잔차로 보정계수 추정
for m in range(3):
    resid = log(true_c[f, m]) - pc_log_fit[f, m]
    smear[m] = np.mean(np.exp(resid))            # 변형: family별 smear[fam, m]
pc = np.exp(pc_log) * smear                       # to_pred() 안에서 적용
# 안전계수 그리드 재스윕 (0.88~0.99로 상한 확장 — 상승을 기대하므로)
```

- **반드시 안전계수 그리드 상한을 넓힐 것.** 이 실험의 이득은 점수가 아니라
  "같은 초과확률에서 더 높은 안전계수"로 나타나므로, 기존 그리드
  (fast 0.90~0.99 / bal 0.84~0.94 / prem 0.78~0.90) 상단에 붙으면 측정이
  잘린다. 각 tier 상한을 1.02까지 확장하고 초과확률도 같이 로깅할 것.
- **export 가능성**: 트리 배열 동일 + 상수 3개(또는 family별 27개). 자명.

### 소요·기대

메타 전용 ~8분/config, (비율 단독 / smear 단독 / 둘 다 / family별 smear)
4구성 = **약 1시간**. 기대 이득 **+0.001 ~ +0.004 EV**, 안전계수가 가장 낮은
**balanced(0.89)·premium(0.88)에 집중** — fast 중심 실험들과 이득이 겹치지
않아 합산 기대가 크다. (a)와 (b)를 반드시 분리 측정해 어느 쪽이 작동했는지
남길 것.

---

## E32. 비용 불확실성 팽창 할당 (문항별 σ̂ 페널티)

- **실행 위치**: Colab CPU (헤드 1~3개 추가 + 사후 그리드)
- **축**: 할당 · EV

### 가설

E08(문항별 비용 상한)이 실패한 이유는 진단이 틀렸기 때문이다 — 예산 초과는
**비싼 소수 문항**이 아니라 **광범위한 비용 오차**에서 온다고 E08이 스스로
결론지었다. 그렇다면 페널티를 걸어야 할 대상은 *비용의 크기*가 아니라
**비용의 예측 불확실성**이다. 이분산성(어떤 문항은 답 길이가 본질적으로
예측 가능하고, 어떤 문항은 아니다)이 존재한다면, 불확실한 문항을 선택에서
밀어내는 것만으로 실현 비용비율의 분산이 줄고 안전계수를 올릴 수 있다.
E08과 페널티의 축이 직교한다.

### 방법

```python
# 1) 이분산 헤드: fold 내부 OOF 잔차의 절댓값을 회귀 (모델별 3개)
resid_oof = np.abs(log(true_c[f, m]) - inner_oof_cost[f, m])
sig_head[m] = HistGBRegressor(**CFG).fit(Xf, resid_oof)
sigma = column_stack([sig_head[m].predict(Xh) for m in range(3)])
# 2) 진단 먼저: sigma 5분위별 실제 |잔차| 평균이 단조 증가하는가?
#    스프레드가 1.3배 미만이면 이분산이 없다는 뜻 → 즉시 기각
# 3) 할당 전용 팽창 비용 (점수 예측·채점에는 쓰지 않는다)
pc_alloc = pc * np.exp(KAPPA * (sigma - sigma.mean(axis=0)))    # KAPPA ∈ {0,.25,.5,1,2}
pick = allocate(ps, pc_alloc, multiplier, safety)                # 실제 비용은 true_c로 채점
# 4) (KAPPA × safety) 2차원 그리드에서 EV 최대점, 초과확률도 함께 기록
```

- `pc_alloc`은 **할당 결정에만** 쓰고 예산 회계·채점은 기존 `pc`/`true_c`로
  한다 (그렇지 않으면 안전계수와 이중 계상된다).
- KAPPA=0에서 기준선 EV가 정확히 재현되는지로 하니스 정합성 검증.
- **export 가능성**: 트리 3개 추가 + 스칼라 KAPPA + 열평균 3개. `math.exp`만
  추가로 필요.

### 소요·기대

헤드 3개 추가로 메타 적합 ~11분/config, (KAPPA×safety) 그리드는 사후처리.
**약 1시간.** 기대 이득 **+0.000 ~ +0.003 EV**. 2단계 진단에서 조기 기각될
확률이 상당하므로(이분산이 약할 수 있음) 그 진단 표를 반드시 로그에 남겨
"비용 오차가 등분산이다"라는 사실 자체를 기록 자산으로 만들 것.

---

## E33. 전역 + family 잔차 계층 트리 (하드 게이트 전문가)

- **실행 위치**: Colab CPU (메타 전용)
- **축**: family 특화 × 메타모델 구조 (트리 계열 내)

### 가설

E13(모듈 사전분포)이 기각된 이유는 **커버리지 10.8%** 와 **kNN이 이미 흡수**
였다. family는 커버리지 100%이고, 여기서 추가하는 것은 사전 평균이 아니라
**모델 용량의 재배분**이다. 현재 전역 GBM은 `max_leaf_nodes=15`로 9개의
이질적 모집단을 하나의 트리 집합으로 처리한다 — family one-hot이 입력에
있어도, 깊이 4의 트리가 "family로 먼저 분기한 뒤 각자 다른 특징으로 3번 더
분기"를 8헤드 전부에 대해 표현할 여유는 없다. family별 잔차 전문가를 얇게
얹으면 그 표현력이 생긴다. 사전분포 주입이 아니라 잔차 학습이라는 점이
E13과의 결정적 차이다.

### 방법

```python
# 1) 전역 8헤드는 현행 그대로 학습 → fold 내부 OOF 예측으로 잔차 산출
resid = y_head - global_oof_pred                       # 8헤드 전부에 대해
# 2) family별 얇은 잔차 전문가 (fold-train 내 n_f >= 150 인 family만)
for fam in FAMILY_NAMES:
    idx = (fam_of[f] == fam)
    if idx.sum() < 150: continue
    exp_f[fam] = HistGBRegressor(max_leaf_nodes=7, learning_rate=0.03,
                                 max_iter=60, min_samples_leaf=25,
                                 early_stopping=True).fit(Xf[idx], resid[idx])
# 3) James-Stein류 수축 가중 (작은 family는 자동으로 전역에 수렴)
w_f = n_f / (n_f + TAU)                                # TAU ∈ {50, 150, 400, 1200}
pred = global_pred + w_f[fam_of[h]] * exp_f[fam_of[h]].predict(Xh)
# 4) 변형: 잔차 전문가를 8헤드 전부가 아니라 δ 헤드 2개에만 적용 (fast 겨냥·경량)
```

- TAU=∞ 가 기준선과 일치해야 한다(정합성 검증). TAU 곡선이 단봉인지 확인.
- 변형 4를 먼저 돌릴 것 — 아티팩트 증가가 1/4이고 fast 겨냥이라 가성비가 높다.
- **export 가능성**: 트리 배열이 family id로 keyed 되어 추가될 뿐이고, 런타임은
  이미 `similarity.classify_family`로 family를 알고 있으며 `evaluate_trees`도
  그대로 쓴다. **다만 heavy 블록이 +20~30% 커질 수 있으므로 QEMU 90초 검사를
  반드시 재측정**할 것 (현재 46~61초로 여유가 크지 않다). 8헤드 전체 변형이
  QEMU를 위협하면 변형 4만 채택하는 경로를 미리 잡아둘 것.

### 소요·기대

메타 전용이지만 전문가 적합이 붙어 ~14분/config, TAU 4점 × 변형 2 =
**약 2시간**. 기대 이득 **+0.000 ~ +0.004 EV**. family별 EV 분해를 함께
출력해, 이득이 특정 family(예: code, longdoc)에서만 나오는지 확인할 것 —
그렇다면 그 family만 선택 적용하는 것이 아티팩트 예산상 유리하다.

---

## 부록: 공통 준수 사항

1. **누출 차단**: 모든 신규 통계량(E28의 토큰 선택, E29의 bin 가중, E30의
   등장성, E31의 smearing, E33의 family 잔차)은 **outer fold-train만으로**
   산출한다. 기존 하니스의 `X_meta_fit` / `X_meta_hold` 분기 규약을 그대로
   따르면 자동으로 지켜진다.
2. **정합성 검증**: 각 실험은 "무효 파라미터"(E27 A=0, E29 GAMMA=0,
   E30 A=0, E32 KAPPA=0, E33 TAU=∞)에서 기준선 EV 0.6976을 **소수 4자리까지
   재현**해야 한다. 재현되지 않으면 결과 해석 전에 하니스를 고칠 것.
3. **판정 기준**: 단일 최고점이 아니라 **파라미터 곡선의 매끄러움·단봉성**으로
   판단한다 (E14 채택 / E21 보류의 판단 근거와 동일). 비단조 곡선의 최고점은
   노이즈로 간주하고, 다른 부트스트랩 시드로 확인 런을 건다.
4. **안전계수 재선택**: 채택되는 모든 변경 후에는 **880 크기** 부트스트랩으로
   tier별 안전계수를 다시 고른다 (2,640 크기 금지 — E11 함정).
5. **기록**: 채택·기각 무관하게 전부 `EXPERIMENT_LOG.md`에 추가한다. 특히
   E30·E32의 진단 표(편향 분위표, 이분산 분위표)는 기각되더라도 이후 설계의
   사전확률을 바꾸므로 수치를 그대로 남긴다.
