# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 이 저장소에서 하는 일

CprE 487/587 Lab 1 (TensorFlow로 Tiny ImageNet CNN 추론·프로파일링·학습) 결과물을 **전체적으로 롤백 체크**하고 개선하는 것이 목적이다.

작업 방식:

1. 사용자가 instruction PDF를 제공한다 (`CprE487_587_Lab1.pdf`).
2. PDF의 요구사항을 섹션 순서대로 **step-by-step**으로 노트북과 대조한다.
3. 각 step에서 문제가 있으면 그 자리에서 고치고, 무엇을 왜 바꿨는지 짧게 보고한 뒤 다음 step으로 넘어간다.

이미 찾아 둔 문제 목록은 `ANALYSIS_KR.md` §3에 있다 (학습 실패 loss=ln(200), EarlyStopping 콜백 미전달, 체크포인트 파일 덮어쓰기, GPU로 측정한 추론, 셀 17 들여쓰기 버그 등). 새로 점검하기 전에 먼저 읽을 것.

## 저장소 / 원격 주의사항

- `origin` = `devjwk/cpre487lab1` (사용자 개인 private 저장소). 개선 작업은 여기에 commit/push한다.
- `upstream` = 팀원 저장소 `zachdixonISU/CPRE-487-Lab01` (원본, main `bb91cf2`를 히스토리째 머지해 옴). **fetch 전용 — `upstream`에는 push하지 말 것.**
- `gh` CLI는 설치되어 있지 않다.
- 이미 제출한 결과물(`Lab01 Report_06.pdf`, `lab1_06.zip`, `lab1_notebook_06.pdf`, `*.pptx`)은 이 저장소에 없고 `../487lab1/`에 있다. 수정·삭제하지 말고 참고용으로만 본다.

## 실행 환경

이 Mac에서는 노트북을 **실행할 수 없다**: Jupyter/TensorFlow가 없고(시스템 Python 3.9.6), `tiny_imagenet` 모듈·`wnids.txt`·`words.txt`·`CNN_TinyImageNet.h5`·`img_data/`가 `.gitignore`로 빠져 있다. 실제 실행은 랩 GPU VM(`cpre587-f26-06`)의 `lab1_venv`에서 한다.

따라서:

- 빌드/린트/테스트 명령은 없다. 로컬 검증은 정적 검토와 `python3 -m json.tool <notebook>` (JSON 유효성) 정도가 전부다.
- 코드 셀을 고쳐도 저장된 출력은 갱신되지 않는다. 출력을 지우거나 지어내지 말고, **VM에서 다시 돌려야 하는 셀 번호**를 사용자에게 알려준다.
- 보고서 수치를 고칠 때는 노트북에 저장된 실제 출력만 근거로 쓴다.

## 구조

노트북이 두 개인데 **`CprE_487_587_Lab1_Template(1).ipynb`가 최종본**이다 (커밋 `bb91cf2`, 셀 36개, 출력 포함). `CprE_487_587_Lab1_Template.ipynb`는 프로파일링·학습이 TODO로 남은 예전 버전이므로 수정 대상이 아니다.

최종본 셀 구성 (0-based 인덱스):

| 셀 | 섹션 | 내용 |
|---|---|---|
| 1 | Tools Setup | import, `ENABLE_GPU` 스위치 |
| 3–8 | Dataset Inspection | Tiny ImageNet 로드, 샘플 3장 시각화, `img_data/image_{i}.bin` 내보내기 |
| 10–14 | Model Loading and Inference | `CNN_TinyImageNet.h5` 로드, Top-1/5/10 정확도 |
| 16–20 | Model Exploration | 필터·feature map 시각화, 가중치/중간 출력 내보내기 |
| 22–27 | Tensorboard | 프로파일러, 단일/online/batch 추론 측정 |
| 29–32 | Training | 콜백 설정, 모델 정의, batch size·epoch 실험 |
| 34–35 | Above and Beyond | 미완성 |

셀들은 위에서 아래로 상태를 공유한다 (`model`, 데이터셋, `callbacks` 등). 한 셀을 고칠 때는 그 변수를 쓰는 뒤쪽 셀도 확인한다.

### 내보내는 바이너리 형식 (Lab 2 C++ 프레임워크가 그대로 읽음)

형식을 바꾸면 `../487lab2/cpre487-587-dnn-framework`의 인덱싱이 깨진다.

- `img_data/image_{i}.bin`: uint8 raw, HWC `(64,64,3)`, 12,288 bytes
- `model_data/<layer>_weights.bin` / `_bias.bin`: float32 raw, Keras 레이아웃 그대로 (Conv `HWIO`, Dense `(in,out)`)
- `img_data/test_input_{i}/<layer>_output.bin`: float32, NHWC `(1,H,W,C)`, flatten 포함 모든 레이어

브랜치 `upstream/inference-profiling`은 main에 머지되지 않았다 (`ENABLE_GPU=False`로 측정한 버전). CPU 측정값이 필요하면 여기서 비교한다.
