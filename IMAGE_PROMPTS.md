# 프롬프트 라우터 설계 공간 — 이미지 생성 프롬프트 모음

목적: "이 문제를 풀 때 각 단계마다 **어떤 선택지들이 있는가**"를 전부 펼쳐 보는 그림.
현재 우리 선택은 각 그림에서 작은 ★ 표시 하나로만 나타내고, 나머지는 동등한 후보로 그립니다.
GPT 이미지 생성에 하나씩 붙여 넣으세요. 공통 스타일 접두어를 매 프롬프트 앞에 붙이면 그림체가 통일됩니다.

---

## 공통 스타일 접두어 (모든 프롬프트 앞에 붙이기)

```
Clean technical infographic, flat vector style, white background, muted blue/orange/gray palette,
labeled boxes and arrows, no photorealism, no decorative clutter, Korean labels allowed, 16:9,
high legibility, textbook diagram quality. Draw OPTIONS as equal-sized cards in a grid;
mark exactly one card per group with a small ★ badge meaning "현재 선택".
```

---

## 0. 설계 공간 전체 지도

```
A "design space map" for a prompt-only LLM router: a horizontal spine of 8 stages —
"입력 표현" → "특징 추출" → "예측기(점수)" → "예측기(비용)" → "결합/스태킹" → "보정" →
"예산 할당" → "런타임/배포". Above each stage a vertical stack of 4–6 option cards
(names only, e.g. 특징 추출: 수작업 특징 / 해시 n-gram / tf-idf / 학습 임베딩 / 규칙·정규식 / 하이브리드).
Each stage's stack has one ★ card. Draw a dashed line threading the ★ cards left-to-right to show
"하나의 파이프라인 = 각 단계에서 카드 하나씩 고른 경로". Caption: "단계별 선택의 조합 = 설계 공간".
```

---

## 1. 문제 정의 단계 — 무엇을 예측할 것인가

### [1-A] 예측 타깃의 선택지
```
Grid of option cards titled "라우터가 예측할 대상": "모델별 절대 점수", "모델 간 점수 차(이득)",
"어느 모델이 최선인지(분류)", "모델 간 순위", "점수 구간 확률(순서형)", "수용 가능 여부(이진)",
"효율(이득/비용)", "직접 최종 선택(정책)". Arrange as a 4×2 grid. Below, a single arrow to a box
"이 선택이 손실 함수·헤드 구조·할당 규칙을 결정". ★ on "점수 구간 확률" and "이득" and "효율" (multi-target).
```

### [1-B] 비용 타깃의 선택지
```
Grid of option cards titled "비용을 어떻게 다룰 것인가": "모델별 고정 단가로 가정", "로그 비용 회귀",
"출력 토큰 길이 예측 후 단가 곱", "비용 분위(상단 꼬리) 예측", "비용 불확실성(분산) 예측",
"비용 비율(모델 간) 예측", "비용 예측 없이 안전계수만". ★ on "로그 비용 회귀".
```

### [1-C] 데이터 사용 방식의 선택지
```
Cards: "train만 학습 / dev 검증", "train+dev 합산 학습 + 교차검증", "K-fold 앙상블",
"공개셋 일부를 held-out 고정", "외부 데이터 증강", "라벨 없는 프롬프트 활용(반지도)".
★ on "train+dev 합산 + 교차검증".
```

---

## 2. 입력 표현 단계 — 텍스트를 무엇으로 볼 것인가

### [2-A] 표현 방식 카탈로그
```
Cards titled "텍스트 → 숫자": "수작업 통계 특징(길이·기호·언어 비율)", "단어 n-gram 해시",
"문자 n-gram 해시", "tf-idf 벡터", "사전학습 문장 임베딩", "소형 트랜스포머 인코더(학습)",
"규칙·정규식 카테고리", "프롬프트 해시 조회(캐시)", "토큰 수 추정치". Show that several can be
used TOGETHER by drawing them feeding one merge node. ★ on 수작업 + 단어 해시 + 문자 해시 + tf-idf +
규칙 + 조회 (multiple ★, caption "조합 사용").
```

### [2-B] 표현의 자유도
```
A single feature-extraction box with orange knobs around it, each knob a dial labeled with a
tunable dimension: "n-gram 크기", "해시 bin 수", "stride/샘플링", "텍스트 길이 한도", "토크나이저",
"정규화(소문자·숫자 치환)", "표준화 방식", "특징 선택 방법", "차원 축소 여부". No values, only names.
```

---

## 3. 점수 예측기 단계 — 어떤 모델 클래스로

### [3-A] 모델 클래스 카탈로그
```
Cards titled "점수 예측기 후보": "선형/릿지 회귀", "로지스틱 회귀", "규칙 기반 hash-regex",
"kNN(유사 문항 평균)", "그래디언트 부스팅 트리", "랜덤 포레스트", "MLP", "트랜스포머 파인튜닝",
"카테고리 평균(family prior)", "행렬 분해(문항×모델)". Group into 3 columns: "표준 라이브러리로
export 가능" (선형·규칙·kNN·트리·평균) vs "런타임 제약상 불가" (MLP·트랜스포머) vs "조건부".
★ on 선형, 규칙, kNN, 트리, 평균 (ensemble).
```

### [3-B] 예측기 조합 방식
```
Cards titled "여러 예측기를 어떻게 합칠까": "고정 가중 평균", "확신도 게이트(유사도 비례)",
"스태킹(2차 모델)", "부스팅 잔차 학습", "tier별 다른 가중", "선택적 스위칭(하나만 고름)",
"베이지안 평균". ★ on 고정 가중 + 확신도 게이트 + 스태킹 + tier별 가중.
```

---

## 4. 헤드 설계 단계 — 타깃마다 어떤 출력을 낼까

### [4-A] 헤드 구성 카탈로그
```
Cards titled "예측 헤드의 종류": "모델별 회귀 헤드", "누적 임계 분류(순서형)", "다중 클래스 분류",
"이득(차이) 회귀", "부호 분해(도움/손해 확률 × 크기)", "쌍별 순위 헤드", "효율 순위 헤드",
"분위 회귀 헤드", "이분산(σ) 헤드", "공유 인코더 + 모델별 헤드". Show them as pluggable slots
into one "메타 입력 벡터" bar. ★ on 회귀 + 순서형 + 이득 + 효율 순위.
```

### [4-B] 헤드 결합의 자유도
```
One "헤드 출력 결합" box with dials: "재구성 혼합 비율(α)", "순위 헤드 혼합(β)", "쌍별 헤드 혼합(γ)",
"tier별 blend", "헤드 간 평균 vs 스태킹", "클램프·단조성 강제 위치". Names only.
```

---

## 5. 보정 단계 — 예측을 얼마나 믿을까

### [5-A] 보정 방법 카탈로그
```
Cards titled "확률·비용 보정 방법": "온도 스케일링", "Platt 스케일링", "등장성 회귀(isotonic)",
"분위별 편향 보정", "Duan 스미어링(로그 재변환)", "family별 보정 상수", "보정 없음(할당기가 흡수)",
"컨포멀 예측 구간". ★ on 보정 없음. Caption: "보정이 필요한지 자체가 실험 대상".
```

---

## 6. 예산 할당 단계 — 배치 안에서 어떻게 나눌까

### [6-A] 할당 규칙 카탈로그
```
Cards titled "예산 안에서 모델을 고르는 규칙": "Lagrangian 효용 최대화(이분탐색)",
"품질 임계값 + 가장 싼 모델", "탐욕 효율순 배차", "정수계획(ILP)", "문항별 비용 상한",
"확률적 배차(샘플링)", "tier별 고정 비율", "임계값 후 잔여예산 재배분(하이브리드)". Show all cards
around a central "배치 예산 게이지". ★ on Lagrangian.
```

### [6-B] 예산 위험 관리의 선택지
```
Cards titled "예산 초과(=0점) 위험을 어떻게 막을까": "전역 안전계수 하나", "tier별 안전계수",
"문항별 비용 팽창(불확실성 비례)", "부트스트랩 EV로 안전계수 선택", "최악 배치 시나리오 기준",
"동적 재해결(실행 중 예산 추적)", "fallback 정책(초과 시 전부 light)". ★ on tier별 안전계수 +
부트스트랩 EV + fallback.
```

---

## 7. 학습·검증 절차 단계

### [7-A] 검증 방식 카탈로그
```
Cards titled "무엇으로 좋고 나쁨을 판단할까": "단일 dev 점수", "K-fold 교차검증", "중첩 CV
(누수 차단)", "부트스트랩 기대점수(예산통과 곱)", "다중 시드 삼각측량", "held-out 재학습",
"분포 이동 스트레스 테스트", "오라클 상한 대비 회수율". ★ on 중첩 CV + 부트스트랩 EV +
다중 시드 + held-out.
```

### [7-B] 채택 규칙의 선택지
```
Cards titled "실험 결과를 언제 반영할까": "기준선 상회 즉시", "여러 시드 전부 상회",
"단봉 곡선 확인", "개선폭 > 노이즈 문턱", "복잡도·크기 비용 감안", "성능 동일 시 단순한 쪽".
★ on 여러 시드 + 단봉 + 문턱.
```

---

## 8. 런타임·배포 단계

### [8-A] 런타임 형태 카탈로그
```
Cards titled "추론이 어디서 어떻게 돌까": "순수 표준 라이브러리 파이썬", "numpy 의존",
"ONNX/경량 런타임", "신경망 프레임워크", "규칙 엔진만", "조회표 + 계산 폴백". Group by
"주최측 컨테이너 제약 통과 가능/불가". ★ on 순수 파이썬 + 조회표 폴백.
```

### [8-B] 아티팩트 구조의 선택지
```
Cards titled "학습 결과물을 어떻게 담을까": "단일 JSON", "본체 + 무거운 블록 분리(지연 로드)",
"압축 바이너리", "트리 → 노드 배열 export", "가중치 양자화", "무결성 해시 검증". ★ on 분리 +
노드 배열 + 해시 검증.
```

---

## 9. 근본 대안 — 파이프라인 자체를 다르게

### [9-A] 아키텍처 패밀리
```
Five large cards side by side, each a mini-architecture sketch with a name:
"A. 규칙 기반 라우터", "B. 특징 + 얕은 학습기 스태킹", "C. 사전학습 인코더 + 헤드",
"D. 문항×모델 행렬 분해/협업 필터링", "E. 강화학습 정책(배치 보상 직접 최적화)".
Under each, two lines: "장점 키워드 / 제약 키워드" (names only, e.g. C: "표현력 / 런타임 불가").
★ on B.
```

### [9-B] 하이브리드 조합의 가능성
```
A matrix: rows = the five architecture families A–E, columns = pipeline stages (표현·예측·결합·할당).
Cells show small icons where a family can be plugged into that stage. Caption: "한 파이프라인 안에서
단계별로 다른 패밀리를 섞을 수 있음". Highlight our path with ★ chain: 표현 A+B, 예측 B, 결합 B, 할당 B.
```

---

## 10. 종합 — 선택지 수와 실험 커버리지

### [10-A] 설계 공간 크기 감각
```
A tree diagram: root "라우터 설계" → 8 stage nodes → each stage fanning into its option cards
(from sections 1–8), drawn small. Colored leaves: green = 실험함·채택, gray = 실험함·기각,
white = 미실험. Legend at the bottom. Purpose: show at a glance which parts of the space were
covered. Do not describe methods; names only.
```

### [10-B] 남은 미탐색 영역
```
Same tree as before but with unexplored (white) leaves enlarged and grouped on the right side
under a header "아직 안 해본 것". Examples as names only: "컨포멀 구간", "ILP 할당",
"행렬 분해", "반지도 학습", "동적 재해결", "분포 이동 스트레스", "RL 정책".
```
