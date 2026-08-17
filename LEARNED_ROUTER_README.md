<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# 학습 라우터: 파이프라인·학습·평가 종합 문서

SK텔레콤 Efficient LLM Routing Challenge 제출용 prompt-only 라우터의 전체
설계 문서다. 공식 저장소 `3cccbf602077a846c13b2cb1356eee1559a631db` 기준이며,
규칙·제출 규격은 상위 작업공간의 `CLAUDE_OFFICIAL_CHALLENGE_HANDOFF.md`와
`docs/`를 따른다. 라우터는 프롬프트 내용만 보고 `ax31-light`, `ax31`,
`axk1-think` 중 하나를 고른다. episode ID, 입력 순서, split 정보는 어떤
특징에도 사용하지 않는다.

## 1. 최종 성능 요약 (2026-08-15)

모든 구성요소를 공개 Train+Dev 합산 2,640문항으로 학습했다. Dev가 학습에
포함되므로 dev self-check(0.7404)는 in-sample 참고치이고, 공식 품질 추정은
**5-fold 중첩 교차검증**이다.

| Tier | CV 기대점수(EV) | 안전계수 | 메타 blend |
|---|---:|---:|---:|
| Fast (가중 0.4) | 0.6647 | 0.98 | 0.60 |
| Balanced (가중 0.3) | 0.6921 | 0.89 | 0.30 |
| Premium (가중 0.3) | 0.7464 | 0.88 | 0.45 |

- **가중 CV 점수 0.6982 / 가중 CV-EV 0.6982** (순서형 점수 헤드 배포, E21;
  gain 헤드 α=0.5 + kNN k=16 위에 적용; 순서형 이전 EV 0.6976, gain 이전
  0.6930/0.6928, k=8 시절 0.6983/0.6966) (공식 hash-regex baseline의
  train-단독 dev 점수 0.6954와 비교할 때, EV는 예산초과=0점 위험까지 반영한
  값임에 유의).[^ord-tier]

[^ord-tier]: 순서형 점수 헤드(E21) 배포 후 tier별(fast/balanced/premium)
  세부 EV·안전계수·blend는 재측정 예정 — 아래 표는 gain 헤드 단계(kNN
  k=16) 값을 그대로 유지한 것이며 최신 가중 수치와 정확히 일치하지 않는다.
- 기대점수(EV) = 부트스트랩 재표집에서의 `점수 × 예산통과확률` 평균.
  비공개 평가는 단 한 번의 표본 추출이고 예산을 넘으면 해당 tier가 0점이므로
  raw 점수가 아니라 EV를 최적화 기준으로 사용한다.
- 부트스트랩은 **880 크기**(공개 Dev 규모)로 재표집한다. 비공개셋 크기가
  비공개이므로 작은 쪽을 가정하는 것이 안전하다 (표본이 작을수록 비용비율
  분산이 커져 초과 위험이 커진다).

## 2. 런타임 예측 파이프라인

한 문항의 예측 (score₃, log-cost₃) 행은 다음 단계로 만들어진다. 모든 혼합은
(score, log-cost) 공간에서 이뤄지며 마지막에 score를 [0,1]로 클램프하고
비용 단조성(light ≤ 3.1 ≤ K1)을 강제한다.

### 2.0 전처리 · 특징 추출

원문 텍스트(또는 multi-turn `messages`의 content를 개행으로 이은 것)에서
세 종류의 특징을 뽑는다 (`learned_router.py`의 `raw_dense_features`,
`feature_items`):

**① 구조 dense 특징 30개** — 텍스트를 직접 훑어 계산:

- 길이 계열: 문자·단어·문장·메시지 수의 log1p, 2k/8k/16k/48k 문자 초과 플래그
- 문자 구성: 한글 비율, ASCII 비율, 구두점 밀도, 숫자 밀도, 평균 단어 길이,
  개행 수, 물음표 수, 숫자 run 수
- 정규식 신호(불리언/카운트): 형식 증명 요구(`prove|theorem|증명|귀류…`),
  프로그램 분석(` ``` `, `def`, `traceback`, `complexity`…), 다중 제약
  (`exactly|at least|반드시|이하…`), 단순 변환(`summarize|번역|요약…`),
  객관식 보기 패턴(`(A)`, `B.`…), 답 형식 제약("answer … only/json/정답만"),
  이야기형 수학(`how many|total|확률…`), 참/거짓·함의 판별, LaTeX
  (`\frac|\sum|$…$`), 표 형태(`|`+다중 개행), JSON/XML 형태

**② 단어 n-gram (8,192 bins)** — 토큰화 정규식
`[A-Za-z]+|[가-힣]+|\d+|[^\w\s]`로 자르고 casefold, 순수 숫자 토큰은
`<number>`로 치환. 유니그램 `w1:tok`과 바이그램 `w2:l\x1fr`을 FNV-1a 64bit로
해시해 최상위 비트로 부호(±1)를 정해 bin에 누적, L2 정규화. 출처·주제
식별의 주력.

**③ 문자 3·4·5-gram (8,192 bins)** — casefold + 공백 압축한 텍스트를 앞뒤
합쳐 최대 6,000자로 자른 뒤 stride 3으로 추출, 같은 방식으로 signed 해싱.
오타·언어 혼용·서식에 강하고 템플릿형 지문 패턴을 잡는다.

이 특징들은 학습 시 저장된 dense 평균/표준편차로 표준화되어 16,414차원
희소 벡터가 된다. kNN용으로는 별도로 `similarity.py`의 해시 문자 tf-idf
(3·4·5-gram, 32,768 bins, sublinear tf, 상위 256 성분)가 계산된다.
family 분류는 내용 정규식만 사용한다 (코드→HRMCR(한국나이/음력)→RuleTaker→
TruthfulQA→Belebele(한글 밀도)→장문(6k+)→AIME($수식$)→DM수학(명령형 시작)→
기타 순의 우선순위 규칙).

```text
프롬프트 텍스트
  ├─ [0] 공개 조회표: SHA-256(텍스트) 히트 시 저장된 tier별 예측을 바로 반환
  │      (공개 Train/Dev 2,640문항; CHALLENGE_RULES가 명시 허용하는 기법)
  ├─ [1] 선형 앙상블: 공식 hash-regex(256-bin) 75% ⊕ 학습 ridge(16,414차원) 25%
  ├─ [2] source-family 평균 혼합 30% (정규식 9-family 분류, 내용 기반)
  ├─ [3] 공개 Train kNN 혼합: 해시 문자 tf-idf 코사인 상위 16이웃의 실제
  │      outcome을 top-1 유사도 × 0.4 가중으로 혼합 (닮을수록 강하게 반영,
  │      안 닮으면 자동으로 무시 → 분포 이탈 시 하방 보호)
  ├─ [4] 스태킹 메타 GBM: 58특징(dense30+family9+legacy6+선형6+kNN7)로 학습한
  │      HistGradientBoosting. 점수 자체는 모델별 누적임계 이진 분류기
  │      **12개**(모델 3 × 임계 P(s≥0.25/0.5/0.75/1) 4)로 E[s]=0.25·Σsigmoid
  │      재구성(순서형 헤드, E21) + **gain 헤드 2개**(업그레이드 이득
  │      s₃.₁−s_light, s_K1−s₃.₁ 직접 회귀; 순서형 s₀ 기준으로 α=0.5 혼합)를
  │      tier별 가중치로 혼합 — 할당이 쓰는 양(이득)을 직접 모델링
  └─ [5] 배치 할당: tier 예산 × 안전계수 아래에서 Lagrangian 이분탐색으로
         문항별 모델 선택
```

- 랭크 효율 헤드(E27): 할당이 쓰는 것은 효율(Δs/Δc)의 순위뿐이라는 관찰에서,
  효율을 fold-train 내 백분위 순위 [0,1]로 변환해 GBM으로 학습하고, 65노드
  분위 LUT로 역변환한 뒤 예측 비용차를 곱해 이득으로 복원, 기존 δ회귀와
  β=0.25로 혼합한다. `meta_rank_trees`는 heavy 블록으로 분리된다.
- 특징: 30개 구조 dense 특징, signed word 1·2-gram 8,192-bin, signed char
  3·4·5-gram 8,192-bin (FNV-1a 해싱). ridge alpha 30.
- 컨테이너는 Python 표준 라이브러리만 사용한다. GBM 트리는 순수 배열로
  export되어 `similarity.evaluate_trees`로 평가된다 (sklearn 예측과 비트
  단위 일치 검증).
- kNN 벡터·타깃과 메타 트리(10.7MB)는 `learned-router-heavy.v1.json`으로
  분리되어 **조회표 미스 시에만** SHA-256 검증 후 지연 로드된다. 본
  artifact는 4.1MB로, 공개 검사에서 무거운 파싱을 전혀 하지 않는다.

## 3. 학습 절차

데이터: `data/combined/inputs.json` + `outcomes.json` (Train 1,760 + Dev 880
materialized 병합, split="public-train-dev"). 재학습 시 아래 순서를 지켜야
한다 (각 단계가 artifact JSON을 누적 수정).

```powershell
# 0) (필요 시) 합산 데이터 재생성 — 병합 로직은 tools/combine_public_data.py 참조
# 1) GPU 선형 head 학습 (CuPy sparse LSMR, RTX 2050에서 수 분)
$env:PYTHONPATH="$PWD\src"
python tools/train_learned_router_gpu.py `
  --input data/combined/inputs.json --outcomes data/combined/outcomes.json `
  --validation-input data/materialized/dev/inputs.json `
  --validation-outcomes data/dev/outcomes.json `
  --artifact src/ossp_router/resources/learned-router.v1.json `
  --report reports/learned-router-gpu-report.v1.json `
  --word-bins 8192 --char-bins 8192 --alphas 30 `
  --blend-weights 0.75 --context-limits 1000000
# 2) family 평균 + kNN 테이블 (idf, 문서벡터 top-256, 5자리 양자화)
python tools/build_router_augmentation.py --artifact src/ossp_router/resources/learned-router.v1.json --train-input data/combined/inputs.json --train-outcomes data/combined/outcomes.json --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
# 3) 메타 GBM (sklearn 필요; 선형 특징은 5-fold OOF, kNN 특징은 LOO로 leakage 차단)
python tools/build_meta_gbm.py --artifact src/ossp_router/resources/learned-router.v1.json --train-input data/combined/inputs.json --train-outcomes data/combined/outcomes.json --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json
# 4) 안전계수: scratchpad의 combined_cv.py(5-fold 중첩 CV + 880 부트스트랩)로
#    tier별 EV 최적값을 구해 artifact tier_safety_ratios에 기록
# 5) 공개 조회표 (tier별 18값)
python tools/build_public_lookup.py `
  --artifact src/ossp_router/resources/learned-router.v1.json `
  --inputs data/materialized/train/inputs.json data/materialized/dev/inputs.json
# 6) 무거운 블록 분리 (kNN 벡터 + 메타 트리 → learned-router-heavy.v1.json)
python tools/pack_artifact.py --artifact src/ossp_router/resources/learned-router.v1.json
```

leakage 차단이 핵심 설계다: 메타 GBM이 소비하는 선형·kNN 특징을 그대로
학습 데이터에서 만들면 in-sample 낙관을 학습해 실패한다 (초기 실험에서
확인). 반드시 out-of-fold / leave-one-out 특징으로 학습한다.

## 4. 평가 방법론

- **5-fold 중첩 CV**: 선형·family·kNN·메타를 모두 fold-train만으로 다시
  만들어 fold-밖 2,640 예측을 조립한 뒤 실제 채점한다 (메타의 선형 특징은
  fold 안에서 다시 내부 OOF).
- **부트스트랩 EV**: CV 예측을 880 크기로 500회 재표집해, 각 표본에서
  할당→실제 비용비율→통과 여부→점수를 계산. `E[점수×통과]`가 EV.
- 안전계수·blend 등 모든 운영 파라미터는 EV 최대화로 선택한다.
- 교훈: 2,640 크기로 재표집하면 분산이 작아져 위험을 과소평가한다. 반드시
  보수적(작은) 표본 크기로 고른다.
- **held-out 검증 (2026-08-16)**: 배포 아티팩트는 Train+Dev 합산으로 학습되어
  Dev를 재채점한 "dev final" 값(예: 0.7286)은 [0]단계 공개 조회표가 Dev를
  이미 암기하고 있어 in-sample이며 성능 지표로 쓸 수 없다. 이를 확인하기
  위해 전체 학습 체인을 **Train 1,760만**으로 재실행하고 조회표에서 Dev를
  제외한 뒤 Dev 880을 순수 held-out으로 채점했다 (`tools/run_holdout.ps1` +
  `tools/holdout_eval.py`, 산출물은 `reports/holdout/`). 결과는 **dev
  0.7000** — CV EV(0.6982)와의 차이가 0.0018에 불과해, 중첩 CV 하니스가
  실제 out-of-sample 성능을 정확히 근사함을 검증한다. 이후 "성능"으로
  인용 가능한 수치는 CV EV(0.6982)와 이 held-out dev(0.7000)뿐이다.

## 5. 성능 개선 이력

| 단계 | 추정 점수 | 비고 |
|---|---:|---|
| 공식 hash-regex baseline | dev 0.6954 | |
| 초기 학습 라우터 (4096-bin, α10, 장문은 legacy-only) | dev 0.6976 | 숨은 예산초과 위험 33% |
| 8192-bin, α30, 전체 길이 앙상블 | dev 0.6985 | |
| + family 평균 + kNN (conf 게이트) | dev 0.7000 | |
| + 교차적합 메타 GBM + tier별 blend | dev 0.6923 (안전 비율) | dev-max 0.7055 |
| + 안전계수 EV 최적화 | EV 0.53 → 0.692 | 초과위험 33%→0.2% |
| + 합산(2,640) 학습 + 아티팩트 분리 | CV 0.6940 / EV 0.6926 | |
| + gain 헤드 (α=0.5) | CV 0.6983 / EV 0.6966 | 스태킹 이후 최대 단일 개선 |
| + kNN k=16 | CV 0.6982 / EV 0.6976 | E20; Colab 스윕으로 발견 |
| + 순서형 점수 헤드 | CV EV 0.6982 | E21; 삼각측량 채택(3-시드 평균 +0.0013), 배포 완료·QEMU 7-8s |
| + 랭크 효율 헤드 β=0.25 (E27) | CV EV(시드7) 0.6976→0.6982(β=0.5 참조치) | E27; 3시드 평균 +0.0007, in-sample 참고치(합산 학습 후 dev 재채점, 성능 지표 아님) 0.7286, QEMU 6.6~7.7s |
| **held-out 검증 (Train-only 재학습, 현재 공식 수치)** | **dev 0.7000 / CV EV 0.6982** | Train 1,760만으로 재학습·조회표에서 Dev 제외 후 Dev 880 순수 채점: fast 0.6741(ratio 1.187)/balanced 0.6955(ratio 1.722)/premium 0.7389(ratio 3.600), 전 tier 예산 통과. baseline(Train-only) dev 0.6954 대비 **+0.0046** |

진행 중: E23 kNN 표현 3종 스윕 (Colab VM2). E22 특징 5종 스윕(Colab VM1)은
완료·전부 기각 (특징 공간 방향 수확 종료, 상세는 §5 아래 및
EXPERIMENT_LOG.md 참조).

모든 실험의 상세 기록(가설·방법·수치·판정)은 **`EXPERIMENT_LOG.md`**에 있다.
새 실험은 채택 여부와 무관하게 반드시 그 파일에 추가한다.

시도했으나 **채택하지 않은** 것: 직접 GBM 라우팅(leakage로 실패, 0.68),
ridge 메타(이득 없음), 문항별 비용 상한 할당(EV 불변), legacy blend
0.85·0.9(하락), DeepMind 모듈 사전분포 혼합(공개 소스 메타데이터 활용;
CV EV +0.0004로 노이즈 수준, 커버리지 10.8%뿐 — kNN이 이미 모듈 신호를
흡수), MLP 메타헤드(E18; 같은 중첩 CV 하니스에서 GBM 대비 단독·혼합 전부
미달), 임베딩 teacher 증류(E25; 다국어 MiniLM kNN teacher 혼합이 λ 증가에
단조 악화 — 해시 문자 n-gram이 신경망 임베딩보다 우수함을 시사), 특징
배터리 5종(E22; dense 확충·word bin 확대·word 3-gram·프리픽스 블록 전부
기준 미달 — E19와 합쳐 특징 공간 방향 수확 종료). dev oracle 대비 잔여
격차는 fast +0.09 수준으로, 선형+해시+kNN 계열의 한계로 판단된다.

## 6. 컨테이너와 런타임 검증

- 공식 `container/Dockerfile` 사용, `linux/arm64`, 표준 라이브러리 전용.
  런타임 모듈: `learned_router.py`, `legacy_hash_regex.py`, `similarity.py`,
  `heuristic.py`, `protocol.py` (+resources). **새 모듈/리소스를 추가하면
  `.dockerignore` 화이트리스트에 반드시 추가할 것.**
- 이 Windows/x86_64 호스트의 QEMU 에뮬레이션 검사(공식 제약 동일: 2 CPU,
  2GiB, read-only, pids 32, 90초/tier)는 공개 2,640문항 기준 tier당 약
  46~61초로 통과. QEMU 시간은 참고값이며 합격/불합격 판단은 네이티브
  ARM64 장비에서 `tools/check_runtime.py`로 해야 한다 (Windows에서는
  `fcntl` 때문에 실행 불가 — 동일 제약을 복제한 대체 스크립트는 세션
  scratchpad `qemu_check.py` 참조).
- 컨테이너와 네이티브의 결정 일치, 아티팩트 분리 전후 결정 동일성을 매번
  확인했다. 순수 Python 추론 경로는 FNV 접두사 캐시·Counter 집계 등으로
  원본 대비 2배 최적화되어 있다 (출력 바이트 동일 검증).

## 7. 주요 파일

- `src/ossp_router/learned_router.py` — 예측 파이프라인 전체(조회표→앙상블→
  family→kNN→메타→할당), artifact 파싱(지연 로드 포함)
- `src/ossp_router/similarity.py` — 해시 tf-idf·kNN 색인·family 분류·트리 평가
- `src/ossp_router/resources/learned-router.v1.json` — 본 artifact (4.1MB)
- `src/ossp_router/resources/learned-router-heavy.v1.json` — 지연 로드 블록 (10.7MB)
- `tools/train_learned_router_gpu.py` / `build_router_augmentation.py` /
  `build_meta_gbm.py` / `build_public_lookup.py` / `pack_artifact.py` — 학습 체인
- `tools/combine_public_data.py` — train+dev 분석용 병합 파일 생성
- `reports/learned-router-cpu-self-check.v1.json` — 공식 scorer 재채점 (in-sample)
- `container/entrypoint.py` — 제출 컨테이너 진입점
