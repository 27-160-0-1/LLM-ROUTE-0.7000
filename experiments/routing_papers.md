# LLM 라우팅 최신 연구 조사 (2025~2026)

조사 배경: SKT Efficient LLM Routing Challenge — 프롬프트만 보고 3개 LLM(소형/중형/추론형) 중 하나를 선택, 배치 비용 예산 제약 하에 평균 품질 최대화. 라벨 2,640개(모델별 성공률+토큰수), 런타임은 **Python 표준 라이브러리 전용**(신경망 배포 불가, 트리/선형/kNN export만 가능). 현재 접근: 해시 n-gram ridge + kNN + GBM 스태킹 + gain 헤드 + Lagrangian 할당 + EV 안전계수(부트스트랩).

---

## 1. Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers (arXiv 2505.12601, 2025)

**핵심 아이디어**: MLP/GNN/attention 기반 학습형 라우터와 단순 kNN을 표준화된 벤치마크에서 직접 비교. 임베딩 공간의 "지역성(locality)" 덕분에 kNN이 학습형 라우터를 자주 능가하며, 특히 **표본 복잡도(sample complexity)가 낮아 소데이터에서 강점**을 보임을 실증.

**적용 가능성**: 매우 높음(이미 부분 채택). 이 챌린지의 kNN 기반 증강(NEIGHBORS=16)이 이 논문의 결론과 정확히 일치 — 소데이터 상황에서 파라메트릭 모델보다 kNN을 신뢰할 근거를 제공. 새로 적용할 점: 논문의 "지역성 진단" 방법론(k별 성능 곡선, 임베딩 공간 클러스터링 품질 체크)을 사용해 현재 k=16이 실제로 국소성 가정에 부합하는지 진단해볼 수 있음.

## 2. Conformal LLM Routing with Distribution-Free Safety Guarantees (ACL 2026 SRW, arXiv 2026.acl-srw.70)

**핵심 아이디어**: 텍스트 임베딩으로 학습한 안전성 게이트에 **Clopper-Pearson 등 conformal 보정**을 적용해, 통계적 분포 가정 없이 "위반율이 확률 1-δ로 α 이하"를 보장하는 라우팅 임계값을 선택. 예산/안전 위반 확률에 대한 형식적 상한을 제공.

**적용 가능성**: 높음. stdlib만으로 quantile 계산이 가능해 즉시 구현 가능. 현재 부트스트랩 기반 EV 안전계수(수동 그리드 탐색)를 **conformal quantile(Clopper-Pearson/경험적 분위수)** 기반 원칙적 캘리브레이션으로 교체하면, "fast 0.98은 그리드 상단"처럼 임의 그리드에 의존하던 부분을 이론적 보장이 있는 절차로 대체할 수 있음.

## 3. CARROT: A Cost Aware Rate Optimal Router (arXiv 2502.03261, ICLR 2025 Workshop)

**핵심 아이디어**: 비용과 정확도를 **각각 별도로 예측**하는 단순한 라우터가 미니맥스 최적(minimax optimal)임을 이론적으로 증명. GPT-4 수준 정확도를 절반 비용으로, 95% 정확도를 20% 비용으로 달성.

**적용 가능성**: 중간(이미 유사 구조 채택). 현재 파이프라인이 이미 (score, cost) 조회표 방식으로 비용·정확도를 분리 예측하고 있어 이론적 정당성 재확인 차원. 새로 적용할 점: 논문의 미니맥스 하한 분석을 참고해 GBM 스태킹의 특징 수(58개)가 과적합 위험 대비 실제로 필요한지 정보이론적 관점에서 재검토 가능.

## 4. IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory (arXiv 2506.01048, ACL 2025)

**핵심 아이디어**: 심리측정학의 **Item Response Theory**를 차용 — 각 LLM을 "잠재 능력(latent ability)" 벡터로, 각 쿼리를 "잠재 난이도(latent difficulty)" 벡터로 모델링하고 이 둘의 내적/로지스틱 관계로 성공 확률을 예측. 소수 파라미터로 해석 가능한 라우팅 수행.

**적용 가능성**: 높음, 신규 아이디어. IRT는 모델당 1~2개, 쿼리당 1~2개 파라미터만 필요해 2,640개 라벨로도 안정적으로 적합 가능하며 순수 로지스틱 함수라 stdlib 구현이 쉬움(트리/선형 export 범주). 현재 GBM 스태킹·kNN과 **직교적인 저차원 특징**(쿼리의 "잠재 난이도" 스칼라)을 만들어 메타 특징에 추가하면 앙상블 다양성을 높일 수 있음.

## 5. Robust Batch-Level Query Routing for Large Language Models under Cost and Capacity Constraints (arXiv 2603.26796, 2026)

**핵심 아이디어**: 쿼리별 그리디 라우팅 대신 **배치 전체를 ILP(정수계획법)로 동시 최적화**, 부트스트랩·conformal prediction으로 모델 성능의 불확실성 분포를 추정해 강건한 할당을 수행. 최소한의 학습 데이터로 작동.

**적용 가능성**: 높음, 신규 아이디어. 현재 Lagrangian 할당은 점별(pointwise) 근사인데, 이 논문처럼 **불확실성 구간을 포함한 배치 단위 정수계획**(stdlib로 단순 DP/그리디 근사 구현 가능, scipy 없이도 knapsack 근사 가능)으로 대체하면 예산 초과 위험을 줄이면서 현재의 과도하게 보수적인 안전계수(fast 0.98)를 완화할 여지가 있음.

## 6. Beyond Query Memorization: LLM Routing with Query Decomposition and Historical Matching (DecoR, arXiv 2605.25558, 2026)

**핵심 아이디어**: 순수 kNN "암기(memorization)" 라우팅은 OOD(분포 밖) 쿼리에서 무작위보다 성능이 나빠질 수 있음을 지적. 쿼리를 **역량 프로파일(스킬/지식영역/난이도 D0-D3)**로 분해한 뒤 이 구조화된 표현으로 매칭해 ID/OOD 모두에서 kNN보다 견고함(ID 89.35%@2.1x vs kNN 86.60%@2.0x, OOD에서 격차 확대).

**적용 가능성**: 중간, 경고성 + 신규 아이디어. 비공개 평가셋이 공개 train/dev와 얼마나 다른지 불확실한 이 챌린지에서 **kNN 과의존의 OOD 리스크**를 직접 경고하는 논문 — 현재 "kNN-off 하한" 분석과 일맥상통. 적용 아이디어: 정규식/키워드 기반 저비용 "난이도 티어" 특징(코드/수학/추론 키워드 존재, 길이, 복잡도 프록시)을 stdlib만으로 추출해 kNN 신뢰도가 낮은(이웃 거리 큰) 쿼리에 대한 폴백 특징으로 추가.

## 7. WISERouter: LLM Routing with Workload Budget Constraint (arXiv 2607.23765, 2026)

**핵심 아이디어**: 라우팅을 **제약 있는 컨텍스트 밴딧(Constrained Contextual Bandit)**으로 정식화, 남은 예산을 스텝별 제약으로 변환하는 **Adaptive Linear Programming(ALP)**으로 쿼리 순서에 따라 동적으로 예산을 배분.

**적용 가능성**: 낮음~중간. 이 챌린지는 온라인 스트리밍이 아니라 배치 전체를 한 번에 예측하는 구조라 밴딧 프레임 자체는 안 맞지만, **ALP의 "잔여 예산 → 다음 결정 임계값 변환"** 아이디어는 현재 정적 Lagrangian乘数 대신 배치 내 순차 재보정(예: 앞쪽 항목이 예산을 많이 쓰면 뒤쪽 임계값 상향)으로 변형 적용 가능.

## 8. Adaptive LLM Routing under Budget Constraints — PILOT (arXiv 2508.21141, EMNLP 2025 Findings)

**핵심 아이디어**: 완전한 쿼리×모델 라벨 없이도 작동하도록 LinUCB를 확장, 예산 제약을 **다중선택 배낭 문제(multi-choice knapsack)**로 모델링. 오프라인 선호 데이터 + 온라인 밴딧 피드백을 결합한 공유 임베딩 공간 사용.

**적용 가능성**: 낮음. 온라인 밴딧 요소는 이 챌린지(오프라인 배치 예측)와 맞지 않으나, **"라우팅 결정 = 다중선택 배낭 문제"** 프레이밍 자체는 현재 EV 최적화와 동일한 골격이라 이론적 벤치마크로 참고할 가치는 있음(신규성 낮음).

## 9. The Routing Plateau: Understanding and Breaking the Accuracy Limits of LLM Routers (arXiv 2606.07587, 2026)

**핵심 아이디어**: 대부분의 학습형 라우터가 일정 정확도 이상에서 정체(plateau)되는 현상을 분석 — 원인으로 (a) 예측 신호와 실제 라벨 노이즈의 근본적 한계, (b) 특징 표현력 부족을 지목하고, 정체를 깨는 데는 새 모델 구조보다 **더 나은 특징/라벨 품질**이 중요함을 보임.

**적용 가능성**: 중간, 진단 참고용. 직접 적용할 알고리즘은 아니지만, 현재 dev 0.70 근처에서 성능이 정체되는 원인이 "모델 용량 부족"이 아니라 "라벨/특징 노이즈"일 가능성을 시사 — 새 모델을 추가하기보다 라벨 신뢰도(토큰수 노이즈, 소표본 성공률 분산)를 다루는 방향에 자원을 재배분하는 근거로 활용 가능.

## 10. UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing (arXiv 2605.18796, 2026)

**핵심 아이디어**: 캐스케이드(단계적 에스컬레이션) 라우팅에서 각 단계 신뢰도를 **캘리브레이션된 불확실성 추정치**로 보정해, 과신(overconfidence)으로 인한 조기 종료 오류를 줄임.

**적용 가능성**: 낮음. 이 챌린지는 캐스케이드(순차 재시도) 구조가 아니라 단발 선택 구조라 직접 이식은 어려움. 다만 **불확실성 캘리브레이션** 자체는 위 2번(conformal routing)과 결합해 gain 헤드의 신뢰구간 추정에 참고 가능.

---

## 적용 우선순위 Top 3

기존 실험(해시 kNN, GBM 스태킹, gain 헤드, EV 안전계수)과 겹치지 않는 **새 아이디어만** 선정:

1. **Conformal/분포무관 예산-위험 캘리브레이션** (Conformal LLM Routing, 논문 2) — 현재 수동 그리드로 튜닝하는 부트스트랩 안전계수를 Clopper-Pearson류 conformal quantile로 교체해, "위반확률 ≤ δ" 형식적 보장을 stdlib만으로 확보. fast tier 안전계수 0.98(그리드 상단)의 불안 요소를 원칙적으로 해소.

2. **IRT 스타일 저차원 잠재난이도/능력 특징** (IRT-Router, 논문 4) — 모델당·쿼리당 소수 파라미터의 로지스틱 잠재변수 모델을 2,640 라벨로 별도 적합해, 기존 kNN·GBM 특징과 직교적인 새 메타 특징으로 추가. 소데이터에 특히 적합하고 stdlib 구현이 간단.

3. **불확실성 포함 배치 단위 정수계획 할당** (Robust Batch-Level Query Routing, 논문 5) — 현재 쿼리별 Lagrangian 근사 할당을, 성능 추정의 신뢰구간을 반영한 배치 전체 knapsack/ILP 근사(stdlib DP)로 대체해 예산 준수와 안전계수 완화를 동시에 노림.
