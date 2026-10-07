# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 이 저장소에서 하는 일

CprE 487/587 Lab 1 (TensorFlow로 Tiny ImageNet CNN 추론·프로파일링·학습)을 **처음부터 새로 실험**한다. 이전에 팀으로 제출한 결과는 `previous_lab_data/`에 묶어 두었고 참고용일 뿐이다 — 수정하지 않는다.

작업 방식:

1. instruction PDF(`CprE487_587_Lab1.pdf`)를 섹션 순서대로 **step-by-step**으로 따라간다. PDF의 노란색 강조 문장이 곧 필수 과제이자 체크 항목이다.
2. 사용자가 랩 머신에서 해당 step을 실행하고 출력을 붙여 준다. 그 출력을 기준과 대조해 통과 / 수정 필요를 판정한다.
3. 오류가 난 부분과 개선점은 그 자리에서 고치고, 무엇을 왜 바꿨는지 짧게 보고한 뒤 다음 step으로 넘어간다.

이전 실험에서 찾은 문제 목록은 `previous_lab_data/ANALYSIS_KR.md` §3에 있다 (학습 실패 loss=ln(200), EarlyStopping 콜백 미전달, 체크포인트 파일 덮어쓰기, GPU로 측정한 추론, 필터 시각화 들여쓰기 버그 등). 같은 실수를 반복하지 않도록 각 step 전에 관련 항목을 확인한다.

## 저장소 / 원격 주의사항

- `origin` = `devjwk/cpre487lab1` (사용자 개인 private 저장소). 개선 작업은 여기에 commit/push한다.
- `upstream` = 팀원 저장소 `zachdixonISU/CPRE-487-Lab01` (원본, main `bb91cf2`를 히스토리째 머지해 옴). **fetch 전용 — `upstream`에는 push하지 말 것.**
- `gh` CLI는 설치되어 있지 않다.
- 제출했던 보고서·zip·pptx(`Lab01 Report_06.pdf` 등)는 이 저장소에 없고 `../487lab1/`에 있다. 참고용으로만 본다.

## 실행 환경

이 Mac에서는 노트북을 **실행할 수 없다** (Jupyter/TensorFlow 없음). 실제 실행은 랩 머신에서 한다:

- 디렉터리: `/home/jwk0425/jwk_personal_lab/lab1/` (PDF 예시의 `~/cpre_487/lab1`이 아님)
- venv: `source lab1_venv/bin/activate` — Python 3.9.25, tensorflow 2.16.2 (PDF 3.2와 일치 확인됨, 2026-10-07)
- `tiny_imagenet` 모듈은 venv에 pip로 설치되어 있고, `CNN_TinyImageNet.h5`·`wnids.txt`·`words.txt`는 랩 머신 디렉터리에만 있다 (`.gitignore` 대상).

따라서 빌드/린트/테스트 명령은 없다. 출력을 지어내지 말고, 판정은 사용자가 붙여 준 실제 출력이나 노트북에 저장된 출력만 근거로 한다.

## 구조

- `CprE487_587_Lab1.pdf` — instruction. 3.1–3.2(환경)는 통과, 3.3 Dataset Inspection부터 진행한다.
- `previous_lab_data/` — 이전 제출본. `CprE_487_587_Lab1_Template(1).ipynb`가 그때의 최종본(커밋 `bb91cf2`, 출력 포함)이고, `(1)` 없는 쪽은 더 오래된 버전이다.
- 새 실험 노트북은 저장소 루트에 둔다.

노트북 섹션 순서는 PDF와 같다: Tools Setup → Dataset Inspection(3.3) → Model Loading and Inference(3.4) → Model Exploration(3.5) → Tensorboard 프로파일링 → Training → Above and Beyond. 셀들은 위에서 아래로 상태를 공유하므로(`model`, 데이터셋, `callbacks` 등) 한 셀을 고칠 때는 그 변수를 쓰는 뒤쪽 셀도 확인한다.

### 내보내는 바이너리 형식 (Lab 2 C++ 프레임워크가 그대로 읽음)

형식을 바꾸면 `../487lab2/cpre487-587-dnn-framework`의 인덱싱이 깨진다.

- `img_data/image_{i}.bin`: uint8 raw, HWC `(64,64,3)`, 12,288 bytes
- `model_data/<layer>_weights.bin` / `_bias.bin`: float32 raw, Keras 레이아웃 그대로 (Conv `HWIO`, Dense `(in,out)`)
- `img_data/test_input_{i}/<layer>_output.bin`: float32, NHWC `(1,H,W,C)`, flatten 포함 모든 레이어

브랜치 `upstream/inference-profiling`은 main에 머지되지 않았다 (`ENABLE_GPU=False`로 측정한 버전). CPU 측정값이 필요하면 여기서 비교한다.
