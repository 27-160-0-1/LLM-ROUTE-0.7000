# 대회 규칙 이해용 이미지 생성 프롬프트 (SKT Efficient LLM Routing Challenge)

목적: 라우팅 대회의 **규칙 자체**를 그림으로 이해하기. 우리 라우터 구조가 아니라
"무엇을 입력받고, 무엇을 결정하고, 어떻게 채점되나"만 다룹니다.
순서대로 보면 한 장씩 개념이 쌓이도록 짰습니다. GPT 이미지 생성에 하나씩 붙여 넣으세요.

---

## 공통 스타일 접두어 (모든 프롬프트 앞에 붙이기)

```
Clean explanatory infographic, flat vector style, white background, muted blue/orange/gray palette,
labeled boxes and arrows, no photorealism, no decorative clutter, Korean labels allowed, 16:9,
high legibility, textbook-diagram quality. Use exact numbers given below verbatim.
```

---

## 1. 등장인물: 모델 세 개와 가격표

```
Three LLM "workers" as simple icons side by side, each with a price tag and a quality badge:
(1) "ax31-light — 싸고 약함: 입력 1 / 출력 4 크레딧 per 1M 토큰, 평균 정답률 0.60",
(2) "ax31 — 중간: 입력 2.1 / 출력 8.5, 평균 정답률 0.68",
(3) "axk1-think — 비싸고 강함: 입력 6.6 / 출력 26.3, 평균 정답률 0.82".
Under the three, one formula card: "문항 비용 = 입력토큰 × 입력단가 + 출력토큰 × 출력단가".
A callout on think: "길게 생각해서 출력 토큰도 몇 배 → 문항당 평균 비용 ≈ light의 23배".
```

## 2. 데이터: 주최측이 미리 다 돌려놨다

```
A table icon showing rows = 문항(prompt), columns = 세 모델. Each cell holds two small numbers:
"점수(0/.25/.5/.75/1)" and "토큰 수". Left of the table, a stack of prompt cards labeled
"공개 train 1,760 + dev 880 = 2,640문항". Right, a locked box labeled "비공개 평가셋 (크기·구성 비공개)".
Caption: "라우터를 만들 때 쓸 수 있는 건 왼쪽뿐. 채점은 오른쪽에서."
```

## 3. 라우터가 보는 것 / 못 보는 것

```
Center: a box labeled "라우터 (참가자 코드)". Left input arrow: a prompt card "프롬프트 텍스트만".
Right output arrow: "모델 하나 선택 (light / mid / think)". Around the router, three crossed-out
items with red X: "모델 답변 본문", "정답", "토큰 수·비용". Footer: "이걸 prompt-only 라우팅이라 부른다:
답을 보기 전에 누구에게 보낼지 정해야 함".
```

## 4. 예산의 정의: '전부 light' 비용이 자(ruler)다

```
A horizontal bar chart. First bar labeled "기준선 = 이 배치 전체를 light로만 처리한 총비용 (=1.0×)".
Three longer bars stacked below it: "fast 한도 = 기준선 × 1.25", "balanced 한도 = 기준선 × 2.0",
"premium 한도 = 기준선 × 4.0". A dotted extra bar far to the right labeled "전부 think ≈ 23× (어느
tier에서도 불가)". Caption: "예산 배수 = light만 쓸 때보다 몇 배까지 써도 되는가".
```

## 5. 같은 배치를 세 번 돌린다

```
One input file icon "비공개셋 N문항" fanning out to three identical router boxes, each with a tier
badge: "--tier fast", "--tier balanced", "--tier premium". Each router outputs its own result file:
"결정표 fast", "결정표 balanced", "결정표 premium" (each = N rows of 문항ID → 모델). Caption:
"같은 라우터, 같은 문항, 예산만 다르게 3회 실행 → 결과표 3장".
```

## 6. 같은 프롬프트, tier마다 다른 결정 (예시)

```
A single prompt card at the top: "예: 적당히 어려운 코드 문제". Three columns below for the tiers.
Each column shows the same three model chips with one highlighted:
fast → "light" highlighted (note: "예산 여유 25%뿐 → 확실한 것만 올림"),
balanced → "mid" highlighted, premium → "think" highlighted (note: "예산 4× → 더 많이 올릴 수 있음").
Second small example row: "AIME 초고난도 수학" → all three columns highlight "light" with note
"think 비용이 너무 커서 4× 예산으로도 감당 불가 → 포기가 정답".
```

## 7. 결정은 문항 하나가 아니라 배치 전체로

```
Left: a table of N prompt rows, each with predicted (점수, 비용) for 3 models. A "예산 게이지" bar
under it. Right: the allocator box "예산 안에서 총점 최대가 되는 조합 선택". Arrows show that a
row's decision depends on the OTHER rows: label "이 문항이 think를 받을지는 절대 난이도가 아니라
같은 배치의 다른 문항 대비 효율(이득/비용)로 정해짐". Caption: "문항 하나짜리 배치는 의미가 다름".
```

## 8. 채점: 점수는 평균, 예산은 절벽

```
Two panels. Left "예산 통과": a check mark, formula "tier 점수 = 선택한 모델들의 점수 평균".
Right "예산 초과 (한 푼이라도)": a big red 0, formula "tier 점수 = 0". Below, a cliff illustration:
x-axis "쓴 비용 / 한도" from 0.9 to 1.1, y-axis "tier 점수"; a flat line up to exactly 1.0 then a
vertical drop to 0. Caption: "비용이 한도와 정확히 같으면 통과. 조금이라도 넘으면 전부 0".
```

## 9. 최종 점수 합산

```
Three tier score cards with weights: "fast × 0.4", "balanced × 0.3", "premium × 0.3", flowing into
"최종 = 0.4·fast + 0.3·balanced + 0.3·premium". Worked example box using real dev numbers:
"fast 0.674 / balanced 0.696 / premium 0.739 → 최종 0.700". A note: "예산 초과한 tier는 0으로 들어감".
```

## 10. 성능 눈금자: 무엇이 좋은 점수인가

```
A single horizontal ruler from 0.60 to 0.85 with ticks: "전부 light 0.605", "공식 baseline 0.695",
"우리 라우터 0.700", "오라클(정답을 다 아는 신) 0.794", "전부 think 0.817 (예산 초과라 실제론 0)".
Shade 0.695–0.794 as "라우팅으로 얻을 수 있는 구간 (폭 0.1)". Caption: "0.80은 오라클보다 높아
불가능; 격차의 대부분은 프롬프트만으로 정답 여부를 알 수 없기 때문".
```

## 11. 왜 예산을 100% 안 쓰나: 안전 마진

```
Two thermometers side by side. Left "라우터가 예측한 비용": filled to "예산의 88%" with label
"안전계수 0.88 — 여기까지만 배차". Right "채점 때 실제 비용": the same batch's real total, drawn as
a fuzzy band around the prediction labeled "예측 오차 (think 출력 길이 ±1.9배)". Show that the band's
top edge sits just under the 100% line. A red example on the far right: "안전계수 1.0으로 꽉 채우면
오차 밴드가 100%를 넘어 → 0점 위험". Caption: "마진 = 통과 시 점수 조금 포기, 대신 0점 절벽 회피".
```

## 12. 트레이드오프 곡선: 기대점수 = 점수 × 통과확률

```
A line chart. x-axis "안전계수 s (0.84 → 0.91)". Three curves: dashed rising line "통과했을 때 점수"
(0.747 → 0.752), dotted rising line "예산 초과 확률" (0.1% → 1.8%, right axis), and a solid hump
"기대점수 EV = 점수 × (1−초과확률)" peaking at s = 0.875 (0.749) then falling to 0.739 at 0.905.
Mark the peak with a flag "최적". Caption: "s를 올리면 점수는 오르지만 0점 확률이 더 빨리 올라
기대값은 떨어진다".
```

## 13. 비공개셋이 다르면 산이 움직인다

```
Same EV-vs-s chart but with three humps overlaid: "공개셋 그대로 (꼭대기 0.875)", "think 출력이 20%
길어진 셋 (꼭대기 0.795)", "배치가 2배 큰 셋 (꼭대기 0.92)". A vertical line at s = 0.88 labeled
"현재 값" crossing the second hump far down its slope, with a red note "이 시나리오면 27% 확률로 0점".
Another vertical line at s = 0.81 labeled "보험 값" that stays near the top of all three.
Caption: "어느 셋이 올지 모르니 최악에서 덜 떨어지는 s를 고른다 (min-regret)".
```

## 14. 실행 환경 제약 (한 장 요약)

```
A container box icon labeled "주최측 도커: linux/arm64" with constraint tags around it:
"Python 표준 라이브러리만 (numpy/torch 없음)", "배치당 90초", "CPU 2 / 메모리 2GB", "네트워크 없음",
"읽기 전용, 비root". Inside the box a small flow: "입력 JSON → 라우터 → 결정 JSON". Footer:
"동점이면 실행 시간 짧은 쪽이 이김". A side note: "공개셋 프롬프트 해시 조회는 허용".
```

## 16. 써도 되는 것 / 쓰면 안 되는 것 (CHALLENGE_RULES §사용할 수 있는 정보·§금지 전략)

```
Two-column card: left column green header "허용", right column red header "금지". Left items:
"공개 train/dev 프롬프트+평가결과로 학습", "공개 비용 계수로 tier 정책 최적화", "분류기·회귀계수·
IDF·토크나이저·조회표·검색색인 이미지에 포함", "정확한 프롬프트/해시로 공개자료 조회", "해시·n-gram·
정규식·임베딩 변환 (=내용 기반 라우팅)", "재배포 가능한 소형 언어모델 (오프라인·자원 한도 내)".
Right items: "세 모델을 순차 호출하거나 답변 비교", "선택 후 재시도·모델 교체·답변 제출",
"challenge_id / split / episode_id / 입력 순서로 정책 변경", "비공개 평가자료·메타데이터 사용",
"평가 중 네트워크·외부 추론 호출", "제출 소스≠이미지, 격리 우회". Footer: "공개셋에서 검증한
동일 프로그램·학습 파일을 최종 평가에도 써야 함".
```

## 17. 실패와 위반은 다르게 처리된다 (§점검과 위반 처리)

```
A decision tree. Root: "tier 실행". Branch A "예산 초과" → "규칙 위반 아님, 해당 tier 점수 0".
Branch B "실행 실패 (시간·메모리·로그·출력 초과, 잘못된 JSON)" → "최대 3회 재실행, 첫 유효 결과 사용,
3회 모두 실패 시 tier 0". Branch C "금지 전략·격리 우회 확인" → "전체 제출 실격". Side note:
"운영자는 문항 ID·순서를 바꾼 입력으로 재실행해 순서 의존성을 점검할 수 있음".
```

## 18. 제출물이란 무엇인가 (§최종 평가와 제출 저장소)

```
Two artifacts side by side: (1) a GitHub repo icon "주최측 리포를 fork한 공개 저장소 — 커밋 SHA
제출, 전체 소스 포함, 평가 종료까지 공개 접근, 수상 시 5년 공개 유지", (2) a container icon
"컨테이너 이미지 — 변경 불가 SHA-256 다이제스트로 특정 (태그 불가)". Between them a badge
"같은 이미지를 세 tier에 각각 실행". Below, a license strip: "코드: Apache-2.0 / MIT / BSD /
ISC / 0BSD (+BSL-1.0, Zlib); 자료: CC-BY-4.0·SA는 귀속 충족 시; copyleft는 사전 승인 필요".
Footer: "최종 점수는 오프라인 계산, 실시간 순위표 없음".
```

## 15. 전체를 한 장으로

```
A single left-to-right storyboard with 6 frames: (1) 세 모델·가격표 → (2) 프롬프트만 보고 선택 →
(3) 배치 전체를 tier별 예산(1.25×/2×/4×)으로 3회 실행 → (4) 예산 안이면 점수 평균, 넘으면 0 →
(5) 0.4/0.3/0.3 가중합 → (6) 안전 마진으로 절벽 회피. Minimal text per frame, arrows between frames.
Title: "LLM 라우팅 챌린지 규칙 한눈에".
```
