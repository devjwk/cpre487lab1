# CPRE-487-Lab01 저장소 분석

출처: https://github.com/zachdixonISU/CPRE-487-Lab01 (2026-09-22에 clone)
팀: Zach Dixon, Jongwoo Kim

## 1. 저장소 구성

| 파일 | 설명 |
|---|---|
| `CprE_487_587_Lab1_Template(1).ipynb` | **최종본** (커밋 `bb91cf2` "last version", 9/4). 셀 36개, 출력 포함 |
| `CprE_487_587_Lab1_Template.ipynb` | 예전 버전 (9/1 머지 시점). 프로파일링/학습 부분이 TODO로 비어 있음 |
| `.gitignore` | `wnids.txt`, `words.txt`, `lab1_venv/`, `img_data/`, `CNN_TinyImageNet.h5` 제외 |
| `README.md` | 제목 한 줄 |

브랜치: `main`, `devjwk-patch-1`(머지됨), `visualization_weights`(머지됨), `inference-profiling`(Zach, **main에 머지 안 됨**).
`inference-profiling`은 최종본과 거의 같고 차이는 `ENABLE_GPU=False`, feature map 컬러맵 `viridis`, 프로파일러 start 부분 try/except 정도.
→ 최종본은 GPU VM(`cpre587-f26-06`)에서 `ENABLE_GPU=True`로 다시 실행한 결과임.

**재현 불가:** `tiny_imagenet` 모듈, `wnids.txt`/`words.txt`, `CNN_TinyImageNet.h5`가 gitignore로 빠져 있어서 이 저장소만으로는 노트북을 실행할 수 없음 (Canvas/랩 머신에서 받아야 함).

## 2. 섹션별 내용과 결과

### Dataset (Tiny ImageNet, 클래스 200개)
- 이미지: `uint8`, `(64,64,3)`, 장당 12,288 bytes
- 샘플 3장(val): kimono(172), gasmask(83), police van(90)
- `img_data/image_{0,1,2}.bin`로 내보냄 — **uint8 raw, HWC 순서, 12,288 bytes**, 메타데이터는 `metadata.txt`

### Model (`CNN_TinyImageNet.h5`)
`conv2d(5x5,32) → conv2d_1(5x5,32) → maxpool → conv2d_2(3x3,64) → conv2d_3(3x3,64) → maxpool → conv2d_4(3x3,64) → conv2d_5(3x3,128) → maxpool → flatten(2048) → dense(256) → dense_1(200, softmax)`
- 파라미터 770,218개 (float32 기준 2.94 MB)
- 추론(샘플 3장): 2/3 정답 (police van을 moving van으로 틀림)
- **val 10,000장 정확도: Top-1 24.30%, Top-5 49.40%, Top-10 61.75%**

### Weight/activation 내보내기 (Lab2 C++에서 사용)
- `model_data/<layer>_weights.bin`, `<layer>_bias.bin` — float32 raw, **Keras 레이아웃 그대로** (Conv는 `HWIO`, 예: `(5,5,3,32)`; Dense는 `(in,out)`, 예: `(2048,256)`)
- `img_data/test_input_{i}/<layer>_output.bin` — 모든 레이어(flatten 포함)의 출력, float32, `(1,H,W,C)` NHWC
- 참고용 가중치 크기: conv2d 9,728 B, conv2d_5 295,424 B / feature map 크기: conv2d 460,800 B, conv2d_5 32,768 B

### Profiling (GPU에서 실행)
| 항목 | 결과 |
|---|---|
| 이미지 1장 지연시간 | 62–66 ms (`model.predict` 호출 1번) |
| Online 10/100/1000장 | 15.2 / 17.6 / 18.5 img/s |
| Batch 20/40/100/200 (1000장) | 238 / 280 / 604 / 2295 img/s |
| 레이어별 시간(µs, 직접 입력한 값) | conv2d_1 490 > conv2d 291 > dense 242 > conv2d_3 156 > conv2d_2 152 … |

### Training (Adam, one-hot, 20 epoch)
| 설정 | Best val Top-1 / Top-5 |
|---|---|
| batch 32 | **학습 안 됨** (loss 5.2984 고정) |
| batch 64 | ≈19.1% / 42.6% (9 epoch 부근, 이후 과적합) |
| batch 128 | ≈22.4% / 47.0% |
| epoch 3 (b32) | 13.09% / 33.20% |
| epoch 10, 100 (b32) | **0.50% / 2.50% — 학습 안 됨** |

## 3. 문제점 (제출/보고서 전에 확인할 것)

1. **학습 실패 (가장 심각).** loss가 5.2984 = ln(200)에 멈춰 있고 정확도가 정확히 1/200, 5/200임 → 모델이 항상 같은 균등분포만 출력하는 상태. ReLU가 모두 죽었을 가능성이 큼 (Adam lr=1e-3, 랜덤 초기화 운). 보고서에서 "epoch을 늘리면 정확도가 떨어진다"고 해석하면 **틀린 결론**. 다시 돌리거나(seed 변경, lr 낮추기) 학습 실패였다고 명시해야 함.
2. **EarlyStopping 미적용.** 셀 29에서 만든 `callbacks`를 `fit()`에 넘기지 않음 → 100 epoch가 끝까지 돌았고(약 22분 낭비) TensorBoard 학습 로그(`train_dir`)도 안 남음.
3. **저장 파일 덮어쓰기.** `CNN_TinyImageNet_train_batch{batch_size}.h5`에 epoch 수가 없어서 3/10/100 epoch 모델이 같은 파일에 덮어써짐.
4. **추론 파트를 GPU로 측정함.** 셀 1 주석에 "inference part는 GPU 끄기"라고 되어 있는데 최종본은 `ENABLE_GPU=True`. 게다가 CUPTI 오류로 GPU tracer가 "0 events"를 수집 → 레이어별 시간 표(셀 25)는 다른 실행(아마 CPU)에서 손으로 옮긴 값이라 나머지 수치와 측정 조건이 다름.
5. **배치 throughput 수치 해석 주의.** 워밍업 없이 batch 20부터 차례로 측정해서 첫 번째 실행이 graph tracing 비용을 떠안음. 한 장씩 `model.predict` 부르는 online 측정(~55 ms/장)은 대부분 호출 오버헤드임 (`model(x)`를 쓰면 훨씬 빠름).
6. **필터 시각화 버그 (셀 17).** `suptitle/tight_layout/show`가 for 루프 **안**에 있어서 2x3 그리드 1장이 아니라 figure 6장으로 나뉨 (출력에 "1 Axes" figure 6개로 확인됨). 한 칸 내어쓰기하면 고쳐짐.
7. **주석과 출력이 다름 (셀 13).** 주석은 24.16% / 61.82%, 실제 출력은 24.30% / 61.75% (CPU→GPU로 다시 돌렸기 때문).
8. **TensorBoard 실행 실패** (exit 255) — 랩 머신에서 직접 `tensorboard --logdir` 실행 필요.
9. 자잘한 것: import 중복, 셀 20에서 `feature_model`을 이미지마다 다시 만듦, 셀 23의 바깥 `for example in ds_train...` 루프 변수를 안쪽에서 다시 씀, 노트북 두 개가 섞여 있음(최종본 이름이 `(1)`), 출력 이미지 때문에 파일이 2.7 MB.

## 4. 다음 랩과의 연결
- Lab2 (`~/487lab2/cpre487-587-dnn-framework`)의 C++ 추론 코드가 위의 `.bin` 파일을 그대로 읽음 → **HWIO / NHWC float32 레이아웃**이 C++ 인덱싱과 맞아야 함.
- Lab3 MAC 유닛은 이 모델의 conv 연산을 int8로 가속하는 것 → 레이어별 시간표에서 conv2d_1(5x5x32x32)이 가장 무거움 (MAC 수 기준으로도 가장 큼).
