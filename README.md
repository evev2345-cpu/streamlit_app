# 기타 코드 & 운지 AI 정밀 분석기 (Streamlit Cloud 배포용)

기타 연주 동영상(MP4, MOV 등) 또는 지판 사진(PNG, JPG 등)을 업로드하면, MediaPipe 21개 3D 손 관절 키포인트 추적, 지판 ROI 전처리, 오디오 FFT 주파수 분석, 최신 Gemini 멀티모달 Vision AI를 통해 실시간으로 정확한 기타 코드명과 6선 가로 벡터 SVG 운지 다이어그램을 생성하는 독립형 웹 애플리케이션입니다.

---

## 1. 파일 구성
- `app.py`: 순수 업로드 기반 Streamlit 메인 웹앱 (유튜브 의존성 완전 제거로 안정성 극대화)
- `guitar_detector.py`: MediaPipe 21개 손 관절 탐지 및 지판 ROI 전처리 모듈
- `audio_dsp.py`: 업로드 동영상에서 ±0.75초 구간 추출 후 12음 크로마그램 및 베이스 근음 분석
- `chord_visualizer.py`: 6선 가로 벡터 SVG 코드 다이어그램 생성 및 AI 응답 파서
- `hand_landmarker.task`: MediaPipe 로컬 CPU 추론 모델 파일
- `requirements.txt`: Streamlit Cloud 파이썬 패키지 목록
- `packages.txt`: Streamlit Cloud 리눅스 시스템 패키지 (`ffmpeg`, `libgl1`, `libglib2.0-0`)

---

## 2. Streamlit Community Cloud (무료 24시간 배포) 방법

1. 깃허브(GitHub)에 새 레포지토리를 생성하고 이 `streamlit_app` 폴더 안의 파일들을 푸시(Push)합니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에 로그인합니다.
3. **'Create app'** 버튼을 누르고 레포지토리를 선택합니다:
   - **Repository**: 사용자의 깃허브 레포지토리
   - **Branch**: `main` 또는 `master`
   - **Main file path**: `app.py`
4. **'Advanced settings'** -> **'Secrets'** 항목에 발급받으신 Gemini API 키를 등록합니다:
   ```toml
   GEMINI_API_KEYS = "여기에_GEMINI_API_키를_입력하세요"
   ```
   *(여러 개의 키가 있는 경우 공백이나 쉼표로 구분하여 입력하면 자동 라운드로빈 및 Quota 장애 전환 작동)*
5. **'Deploy'**를 누르면 자동으로 배포가 완료되며 고유 웹 주소가 생성됩니다.

---

## 3. 로컬 실행 방법 (테스트용)

```powershell
pip install -r requirements.txt
streamlit run app.py
```
