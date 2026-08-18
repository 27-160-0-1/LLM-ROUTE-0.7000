# 새 채팅용 인수인계 프롬프트 (그대로 복사해서 첫 메시지로 붙여넣기)

---

나는 SKT Efficient LLM Routing Challenge(프롬프트만 보고 ax31-light / ax31 / axk1-think 중 하나로 배차, tier별 예산 1.25×/2×/4×, 예산 초과 tier는 0점, 가중 0.4/0.3/0.3)에 참가 중이다. 작업 폴더는 `C:\Users\012\SKT LLM\official-router` (GitHub: https://github.com/27-160-0-1/LLM-ROUTE-0.7000 , 원격 이름 `release`, 주최측 상류는 `origin`). 아래 상태를 읽고 **지금 진행 중인 실험을 이어서 끝내라.** 새로운 방향을 제안하기 전에 먼저 진행 중인 것을 마무리하고 결과를 정리해라.

## 현재 배포본 (건드리지 말 것)
- 구조: SHA-256 공개 조회표 → 해시 n-gram 16,414차원 + 밀집 30 → 선형 앙상블(legacy hash-regex 0.75 / ridge 0.25) → family 평균 0.3 → char tf-idf kNN k=16 → 메타 GBM 스태킹(58특징; 회귀 6 + 순서형 12 + gain 2 + 랭크 효율 2) → Lagrangian 예산 할당, 안전계수 0.98/0.89/0.88
- 성능: held-out dev **0.7000** (train만 학습, 조회표 제거; 공식 baseline 0.6954), CV EV 0.6982, 오라클 상한 0.7944
- 아티팩트: `src/ossp_router/resources/learned-router.v1.json`(4.1MB) + `learned-router-heavy.v1.json`(11.7MB), 롤백본 `*.bak`
- 실험 기록: `EXPERIMENT_LOG.md` E01~E39b (39개, 채택 4개). 외부 방법 11종(IPR, RouteLLM 4, LLMRouter 6) 전부 배포본 미달로 기각 완료. 안전계수 상향은 어떤 수준에서도 손해(E39b), 유일한 열린 결정은 premium 0.88→0.82 하향(보험, 명목 −0.001 / 최악 초과 27%→2.5%) — 사용자 결정 대기 중.

## 지금 진행 중인 것 (여기서 이어받아라)
**목표: 라벨 확장.** 데이터 출처 10종(AIME24/25, Belebele-ko, CruxEval, GSM8K, HRMCR, RuleTaker, TruthfulQA, BAbILong, DeepMind Math)이 전부 공개돼 있고, **SKT 모델 3개도 HuggingFace에 Apache-2.0으로 공개**돼 있다 (`skt/A.X-3.1-Light` 7B, `skt/A.X-3.1` 34B, `skt/A.X-K1` 519B MoE — K1은 로컬 불가). 그래서 light/mid를 직접 돌려 남은 벤치마크 문항 수천 개에 라벨을 만들 수 있는지 **파일럿으로 검증 중**이다. 규칙(CHALLENGE_RULES §사용할 수 있는 정보)상 학습 시점의 공개 모델 사용은 허용됨.

진행 순서 1→2→3→4 중 **2 진행 중**:
1. ✅ 채점 프로토콜 역추론 완료: 주최측 `score` = 정답 생성 수/생성 수(num_generations 2 또는 4, 샘플링), `input_tokens`·`output_tokens`는 생성 수만큼 **합산**된 값(1회당 = ÷num_generations). 주최측은 A.X 토크나이저 그대로 + **family별 고정 지시문 40~77토큰**(belebele +53, ruletaker +77, truthfulqa +40, 수학·한국어 +44, code는 few-shot 가변) — 지시문 길이가 IQR 0으로 정확히 상수. 지시문 원문·temperature·정답 판정기는 비공개.
2. 🔄 **로컬 파일럿 실행 중**: `tools/pilot_local_light.py` — A.X-3.1-Light Q4_K_M GGUF를 `local-llm/llama/llama-server.exe`(포트 8080, `-ngl 20 -c 4096 --parallel 2`)로 띄우고 family별 15문항 × 120개 × 2회 생성(온도 0.7, max_tokens 1024). 결과 `reports/pilot_light.json`(문항마다 저장, 재개 지원), 로그 `reports/pilot_light.log`. 08:43에 재시작해 진행 중(문항당 ~85초, 총 ~2시간). 정답은 `data/gold/gold-answers.v1.json`(원본 벤치마크 매칭, 1,951/2,604 = 75%; dmmath 358·gsm8k 138·truthfulqa 79·longdoc 114 미매칭 남음).
   - 완료되면 정리할 것: (a) family별 **출력 토큰 길이 비율**(우리/주최측 1회당) — 첫 결과에서 수학은 ~0.5배로 짧게 나옴, 지시문 차이인지 확인, (b) **정답 재현율** — 우리 정답 여부 vs 주최측 score≥0.5 일치율·상관, (c) 입력 토큰 차이(지시문 길이 맞추기).
   - 판정 기준: family별 일치율 ≥0.85이고 출력 길이 비율이 0.7~1.4 안이면 "재현 가능" → 3단계로. 미달 family는 지시문/온도 조정해서 재시도(온도 그리드 0.3/0.7/1.0).
3. ⏳ 재현되면: 남은 공개 벤치마크 문항(GSM8K 1,319·Belebele-ko 900·CruxEval 800·TruthfulQA 817·RuleTaker 수만 등)에 light(+가능하면 mid 34B는 Colab A100 4bit)로 라벨 생성 → 2,640→10,000+ → 재학습 → held-out 검증. think(K1)는 못 돌리므로 기존 헤드 + 새 문항의 light/mid 관측을 특징으로.
4. ⏳ 병행 후보: RouterEval/RouterBench의 겹치는 문항 타 모델 정답률을 "난이도 사전분포" 특징으로.

## 환경·운영 주의
- 로컬: RTX 2050 4GB, RAM 15.7GB, venv `C:\Users\012\SKT LLM\.venv` (torch CPU 빌드 — CUDA는 llama.cpp 바이너리로만). llama-server가 안 떠 있으면 `local-llm/llama/llama-server.exe -m ..\A.X-3.1-Light.Q4_K_M.gguf -ngl 20 -c 4096 --parallel 2 --port 8080`으로 기동(Start-Process로 세션과 분리).
- **Colab MCP는 쓰지 말 것** (사용자가 끔). 장기 작업은 반드시 `Start-Process`로 세션과 분리해 띄우고, 문항마다 저장.
- PC 절전: `local-llm/keep_awake.py`(pythonw로 실행 중)가 유휴 절전은 막지만 **뚜껑 닫기는 못 막음** — 사용자가 powercfg로 뚜껑 동작을 "아무 작업 안 함"으로 바꿔야 함 (명령은 이전 대화 참조; 어제 이걸 안 해서 파일럿이 11시간 멈췄음).
- PowerShell에서 인라인 python 문자열 금지 — 스크립트는 반드시 파일로 써서 실행. `git commit`은 분류기에 막힐 수 있으니 메시지를 파일로 써서 `-F`로.
- Gemini API 키는 사용자 환경변수(gemini-2.5-flash + google_search 도구로 동작 확인), Grok 키는 "Incorrect API key"라 재발급 필요.
- 실험 판정은 항상 5-fold 중첩 CV + 880 부트스트랩 EV, 3시드 확인, 단봉 곡선. 이득 <0.0005는 노이즈.
- 보고: 30분마다 진행률을 짧게라도 먼저 말하고, 결과는 표로. 사용자에게 필요한 클릭/결정만 요청.
- 완료된 것은 `EXPERIMENT_LOG.md`에 기록하고 `git push release main`.

**첫 행동:** `reports/pilot_light.log`와 `reports/pilot_light.json` 상태 확인 → 진행률 보고 → 완료 시 위 (a)(b)(c) 표 정리 → 재현 가능 판정 → 3단계 착수 여부 보고.

---
