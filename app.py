import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time

import cv2
from PIL import Image
import requests
import streamlit as st
import streamlit.components.v1 as components

import audio_dsp
import chord_visualizer
import guitar_detector

# ==============================================================================
# 1. API 키 설정 (Streamlit Secrets 또는 환경 변수에서 안전하게 로드)
# ==============================================================================
DEFAULT_API_KEYS = []
_env_keys = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")

if not _env_keys:
    try:
        sec = getattr(st, "secrets", None)
        if sec:
            if "GEMINI_API_KEYS" in sec:
                raw_sec = sec["GEMINI_API_KEYS"]
                if isinstance(raw_sec, list):
                    DEFAULT_API_KEYS.extend([str(k).strip() for k in raw_sec if str(k).strip()])
                else:
                    _env_keys = str(raw_sec)
            elif "GEMINI_API_KEY" in sec:
                _env_keys = str(sec["GEMINI_API_KEY"])
    except Exception:
        pass

if _env_keys.strip():
    for _k in re.split(r"[\n, ]+", _env_keys.strip()):
        if _k.strip() and _k.strip() not in DEFAULT_API_KEYS:
            DEFAULT_API_KEYS.append(_k.strip())

_KEY_LOCK = threading.Lock()
_KEY_ROTATION_IDX = 0
_KEY_COOLDOWN: dict[str, float] = {}

# 무료 고성능 멀티모달 모델 우선순위
FAST_FREE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

# ==============================================================================
# 2. 페이지 레이아웃 및 다크/라이트 테마
# ==============================================================================
st.set_page_config(
    page_title="이 코드 뭔가요? - 기타 코드 AI 정밀 분석기",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"
if "target_time" not in st.session_state:
    st.session_state["target_time"] = 0.0

is_light_mode = (st.session_state["theme_mode"] == "light")

BULB_ON_SVG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23F59E0B'%3E%3Cpath d='M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7M9 21a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-1H9z'/%3E%3C/svg%3E"
BULB_OFF_SVG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2364748B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 18h6'/%3E%3Cpath d='M10 22h4'/%3E%3Cpath d='M12 2a7 7 0 0 0-7 7c0 2.6 1.4 4.8 3.5 6v1a1 1 0 0 0 1 1h5a1 1 0 0 0 1-1v-1c2.1-1.2 3.5-3.4 3.5-6a7 7 0 0 0-7-7z'/%3E%3C/svg%3E"

def _toggle_theme_mode():
    st.session_state["theme_mode"] = "dark" if st.session_state["theme_mode"] == "light" else "light"

if is_light_mode:
    theme_vars = f"""
        --app-bg: #FFFFFF;
        --app-text: #1E293B;
        --title-color: #0F172A;
        --sub-color: #475569;
        --toggle-bg: #F8FAFC;
        --toggle-text: #475569;
        --toggle-border: #E2E8F0;
        --toggle-hover-bg: #F1F5F9;
        --toggle-hover-border: #CBD5E1;
        --bulb-svg: url("{BULB_ON_SVG}");
        --bulb-shadow: drop-shadow(0 0 6px rgba(245, 158, 11, 0.7));
        --input-bg: #F8FAFC;
        --input-text: #1E293B;
        --input-border: #CBD5E1;
        --num-btn-bg: #E2E8F0;
        --num-btn-text: #1E293B;
        --primary-btn-bg: #2563EB;
        --primary-btn-hover: #1D4ED8;
        --primary-btn-border: 1px solid #1D4ED8;
        --alert-bg: #F8FAFC;
        --alert-text: #1E293B;
        --alert-border: #E2E8F0;
        --uploader-bg: #F8FAFC;
        --uploader-border: #CBD5E1;
        --uploader-text: #475569;
        --uploader-subtext: #64748B;
        --uploader-icon: #64748B;
    """
else:
    theme_vars = f"""
        --app-bg: #0F1720;
        --app-text: #F7E7D7;
        --title-color: #FFFFFF;
        --sub-color: #CBD5E1;
        --toggle-bg: #162230;
        --toggle-text: #94A3B8;
        --toggle-border: #2B4E73;
        --toggle-hover-bg: #1B3A57;
        --toggle-hover-border: #4A7A9E;
        --bulb-svg: url("{BULB_OFF_SVG}");
        --bulb-shadow: none;
        --input-bg: #162230;
        --input-text: #F7E7D7;
        --input-border: #2B4E73;
        --num-btn-bg: #1B3A57;
        --num-btn-text: #F7E7D7;
        --primary-btn-bg: #1B3A57;
        --primary-btn-hover: #244C73;
        --primary-btn-border: 1px solid #2B4E73;
        --alert-bg: #162230;
        --alert-text: #F7E7D7;
        --alert-border: #1B3A57;
        --uploader-bg: #162230;
        --uploader-border: #2B4E73;
        --uploader-text: #94A3B8;
        --uploader-subtext: #64748B;
        --uploader-icon: #94A3B8;
    """

st.markdown(f"""
<style>
:root {{
    {theme_vars}
}}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    display: none !important;
}}
.block-container {{
    padding: 1.5rem 2.5rem 2.5rem 2.5rem !important;
    max-width: 100% !important;
}}
.stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--app-bg) !important;
    color: var(--app-text) !important;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}}
h1, h2, h3, h4, h5, h6, p, label, span {{
    color: var(--app-text);
}}
.banner-container {{
    border: 2px solid #3B82F6;
    border-radius: 12px;
    padding: 1.1rem 1.6rem;
    margin-bottom: 1.25rem;
    background: transparent;
}}
.banner-title {{
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--title-color);
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}}
.banner-sub {{
    font-size: 0.95rem;
    color: var(--sub-color);
    font-weight: 500;
}}
.st-key-theme_switch_btn {{
    margin-bottom: 0.6rem !important;
}}
.st-key-theme_switch_btn button {{
    background-color: var(--toggle-bg) !important;
    color: var(--toggle-text) !important;
    border: 1px solid var(--toggle-border) !important;
    border-radius: 999px !important;
    padding: 6px 16px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}}
.st-key-theme_switch_btn button * {{
    color: var(--toggle-text) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}
.st-key-theme_switch_btn button::before {{
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    background-image: var(--bulb-svg);
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    filter: var(--bulb-shadow);
}}
.st-key-theme_switch_btn button:hover {{
    background-color: var(--toggle-hover-bg) !important;
    border-color: var(--toggle-hover-border) !important;
    transform: translateY(-1px);
}}
div[data-testid="stButton"] > button[kind="primary"] {{
    background-color: var(--primary-btn-bg) !important;
    color: #FFFFFF !important;
    border: var(--primary-btn-border) !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 8px !important;
}}
div[data-testid="stButton"] > button[kind="primary"]:hover {{
    background-color: var(--primary-btn-hover) !important;
}}
.chord-card {{
    background: #FFFFFF;
    border-radius: 10px;
    padding: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid #CBD5E1;
}}
div[data-testid="stFileUploader"] {{
    background-color: var(--uploader-bg) !important;
    border: 1px dashed var(--uploader-border) !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
}}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label p,
div[data-testid="stFileUploader"] label span {{
    color: var(--uploader-text) !important;
    font-weight: 600 !important;
}}
div[data-testid="stFileUploader"] section {{
    background-color: transparent !important;
}}
div[data-testid="stFileUploader"] section * {{
    color: var(--uploader-text) !important;
}}
div[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {{
    color: var(--uploader-text) !important;
}}
div[data-testid="stFileUploader"] small {{
    color: var(--uploader-subtext) !important;
}}
div[data-testid="stFileUploader"] svg,
div[data-testid="stFileUploader"] svg * {{
    color: var(--uploader-icon) !important;
    fill: var(--uploader-icon) !important;
    stroke: var(--uploader-icon) !important;
}}
div[data-testid="stFileUploader"] button {{
    background-color: var(--toggle-bg) !important;
    color: var(--uploader-text) !important;
    border: 1px solid var(--toggle-border) !important;
    border-radius: 6px !important;
}}
div[data-testid="stFileUploader"] button * {{
    color: var(--uploader-text) !important;
}}
</style>
""", unsafe_allow_html=True)

# 상단 좌측: 테마 토글 (전구 아이콘 버튼)
theme_col, _ = st.columns([2.5, 7.5])
with theme_col:
    theme_label = "다크모드로 전환" if is_light_mode else "라이트모드로 전환"
    st.button(
        theme_label,
        key="theme_switch_btn",
        on_click=_toggle_theme_mode,
    )

# 메인 배너: 파란색 테두리 전폭 배너
st.markdown("""
<div class="banner-container">
    <div class="banner-title">이 코드 뭔가요?</div>
    <div class="banner-sub">사운드 & 비전 기타 코드 AI 정밀 분석기</div>
</div>
""", unsafe_allow_html=True)

# API 키 입력 지원 (환경변수나 Streamlit secrets가 설정되지 않은 경우)
if not DEFAULT_API_KEYS:
    with st.expander("Gemini API Key 설정 (현재 키 미등록)", expanded=True):
        st.info("Streamlit Community Cloud 배포 시 Secrets 설정에 GEMINI_API_KEYS를 등록하거나, 아래 입력창에 API 키를 입력해 주세요.")
        user_key = st.text_input("Gemini API Key 입력", type="password", key="user_api_key_input")
        if user_key.strip():
            for _k in re.split(r"[\n, ]+", user_key.strip()):
                if _k.strip() and _k.strip() not in DEFAULT_API_KEYS:
                    DEFAULT_API_KEYS.append(_k.strip())

# ==============================================================================
# 3. 로컬 비디오/오디오 프레임 추출 함수
# ==============================================================================
def extract_full_frame_fast(video_path: str, timestamp_sec: float) -> Image.Image | None:
    target_t = max(0.0, float(timestamp_sec))
    coarse_seek = max(0.0, target_t - 1.0)
    fine_seek = target_t - coarse_seek
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{coarse_seek:.3f}",
            "-i", video_path,
            "-ss", f"{fine_seek:.3f}",
            "-frames:v", "1",
            "-q:v", "2",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=8)
        if proc.returncode == 0 and proc.stdout:
            return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, target_t * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
    except Exception as e:
        print(f"프레임 추출 오류: {e}")
    return None


def extract_audio_clip_fast(video_path: str, timestamp_sec: float) -> bytes:
    start_time = max(0.0, float(timestamp_sec) - 0.75)
    duration = 1.50
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=8)
        if proc.returncode == 0 and len(proc.stdout) > 1000:
            return proc.stdout
    except Exception:
        pass
    return b""


def _pil_to_b64_jpeg(pil_img: Image.Image, max_dim: int = 1024, quality: int = 88) -> str:
    buf = io.BytesIO()
    img_copy = pil_img.copy()
    img_copy.thumbnail((max_dim, max_dim))
    img_copy.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_gemini_vision_first_fast(
    api_keys: list[str] | str,
    full_frame_pil: Image.Image,
    clean_zoom_pil: Image.Image,
    annotated_zoom_pil: Image.Image,
    prompt_text: str,
):
    global _KEY_ROTATION_IDX
    raw_keys = [api_keys] if isinstance(api_keys, str) else list(api_keys)
    keys = [k.strip() for k in raw_keys if k and k.strip()]
    if not keys:
        raise ValueError("등록된 Gemini API Key가 없습니다. 상단 설정에서 API 키를 입력해 주세요.")

    b64_full = _pil_to_b64_jpeg(full_frame_pil, max_dim=960, quality=82)
    b64_clean = _pil_to_b64_jpeg(clean_zoom_pil, max_dim=960, quality=92)
    b64_annot = _pil_to_b64_jpeg(annotated_zoom_pil, max_dim=800, quality=88)

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_full}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_clean}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_annot}},
                    {"text": prompt_text},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.05,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    now = time.time()
    with _KEY_LOCK:
        active_keys = [k for k in keys if _KEY_COOLDOWN.get(k, 0.0) < now]
        if not active_keys:
            active_keys = keys
        n_keys = len(active_keys)
        start_idx = _KEY_ROTATION_IDX % n_keys
        _KEY_ROTATION_IDX += 1
        ordered_keys = [active_keys[(start_idx + i) % n_keys] for i in range(n_keys)]

    http_session = requests.Session()
    last_err = ""
    for current_key in ordered_keys:
        key_masked = f"{current_key[:12]}..."
        for model_name in FAST_FREE_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={current_key}"
            try:
                resp = http_session.post(url, json=payload, timeout=9)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return text, f"{model_name} (Key: {key_masked})"
                elif resp.status_code == 429:
                    with _KEY_LOCK:
                        _KEY_COOLDOWN[current_key] = time.time() + 60.0
                    err_msg = resp.json().get("error", {}).get("message", "429 Quota Exceeded")
                    last_err = f"[{key_masked} / {model_name}] {err_msg}"
                    break
                else:
                    err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                    last_err = f"[{key_masked} / {model_name}] {err_msg}"
                    continue
            except Exception as e:
                last_err = f"[{key_masked} / {model_name}] {e}"
                continue

    raise RuntimeError(f"모든 등록된 API 키 및 모델 호출 실패: {last_err}")


def build_vision_first_prompt(hand_section: str, dsp_section: str) -> str:
    return (
        "당신은 일렉기타 및 어쿠스틱 기타 지판 운지 분석 최고 전문가입니다.\n"
        "- 이미지 1: [전체 연주 화면/사진] (기타 헤드스톡과 바디 방향, 카포 유무 확인용)\n"
        "- 이미지 2: [운지 손 & 지판 고해상도 클로즈업] (프렛 와이어, 인레이 점, 실제 줄 접촉 확인용)\n"
        "- 이미지 3: [MediaPipe 손가락 번호 마커 가이드 또는 지판 ROI] (빨강 1=검지, 초록 2=중지, 파랑 3=약지, 보라 4=새끼, 주황 원=엄지 끝)\n"
        f"{hand_section}"
        f"{dsp_section}\n"
        "[절대 준수 규칙 — 시각(Vision) 일치 최우선]\n"
        "1. **헤드스톡 방향 및 프렛 번호 판별**:\n"
        "   - 기타 헤드스톡(너트)이 오른쪽(right)에 있는지 왼쪽(left)에 있는지 확인하세요.\n"
        "   - 헤드스톡이 오른쪽에 있으면 오른쪽 끝(너트)이 0/1프렛이고, 왼쪽으로 갈수록 2프렛, 3프렛(인레이 점), 4프렛 순으로 프렛 번호가 증가합니다.\n"
        "2. **줄 번호 물리적 방향 (1번줄 vs 6번줄 혼동 금지)**:\n"
        "   - 지판의 **위쪽(천장/엄지 방향) 가장 두꺼운 줄이 6번 줄(Low E)**입니다.\n"
        "   - 지판의 **아래쪽(바닥/손바닥 방향) 가장 얇은 줄이 1번 줄(High E)**입니다.\n"
        "3. **카포(Capo) 장착 시 프렛 표기 및 코드명 절대 규칙**:\n"
        "   - 기타 넥에 카포(Capo)가 물려 있는지 꼼꼼히 확인하세요 (예: 1프렛, 2프렛 등).\n"
        "   - 카포가 있는 경우:\n"
        "     * **물리적 프렛 (dots의 fret)**: 반드시 너트(헤드스톡)로부터 세는 **실제 물리적 프렛 번호(절대 프렛)**를 입력하세요.\n"
        "       (예: 1프렛에 카포가 물려 있고 연주자가 카포 뒤 2번째 칸을 누른다면 dots의 fret은 3입니다: 1 + 2 = 3프렛)\n"
        "     * **최종 코드명 (chord)**: 카포로 인해 이조된 **실제 울리는 음향적 코드(Concert Pitch)**를 표기하세요.\n"
        "       (예: 카포 1프렛에서 Am7 오픈 폼을 잡고 있다면 실제 울리는 소리는 **Bbm7** 또는 **Am7 (Capo 1)** 입니다)\n"
        "     * **운지 설명 (fingering_summary)**: '카포 N프렛 장착, [상대 운지 형태] 폼 연주'임을 명확히 서술하세요.\n"
        "4. **실제 줄을 누르고 있는 활성 손가락만 카운트**:\n"
        "   - 4개 손가락 중 **실제로 프렛 지판 줄을 짚고 있는 손가락만** 점(dots)에 포함하세요.\n"
        "   - 새끼손가락(4번)이나 다른 손가락이 넥 아래로 웅크려 있거나 공중에 떠서 줄을 누르지 않는 경우(예: E, Em, C, Am 등), 절대 억지로 4손가락 코드로 왜곡하지 마세요!\n"
        "5. **오픈 E vs Am vs B7 엄격 판별**:\n"
        "   - 만약 1프렛(검지) 1개 손가락, 2프렛(중지, 약지) 2개 손가락으로 총 3개 손가락만 줄을 누르고 있다면:\n"
        "     * 5번줄 2프렛(중지), 4번줄 2프렛(약지), 3번줄 1프렛(검지)를 누르고 6, 2, 1번줄이 개방현이면 100% **E (E Major)** 코드입니다! (절대 B7이 아님!)\n"
        "     * 4번줄 2프렛, 3번줄 2프렛, 2번줄 1프렛이면 **Am** 코드입니다.\n"
        "     * B7 코드는 4개 손가락(5th-2, 4th-1, 3rd-2, 1st-2)이 모두 줄을 눌러야 하므로, 3손가락만 누르고 있으면 절대 B7이 아닙니다.\n"
        "6. **바레 vs 독립 운지 구분**: 검지(1번, 빨강)가 여러 줄을 일자로 길게 누르는 바레(Barre)인지, 한 줄만 짚는 독립 운지인지 이미지 2와 3을 보고 정확히 판별하세요.\n"
        "7. **재즈/얼터드 텐션 코드(7#5, 7b13, m7b5 등) 운지 정확성**:\n"
        "   - G7(#5) 등 6번줄 루트 7(#5) 보이싱은 검지가 6번줄 N프렛(근음)과 4번줄 N프렛(단7도)을 짚고, 중지와 약지가 N+1프렛의 3번줄(3도)과 2번줄(#5도)을 누르며, 새끼손가락은 손바닥에 웅크려 사용하지 않습니다 (5번줄, 1번줄 뮤트). 절대 4번줄을 N+1프렛으로 판정하지 마세요.\n"
        "   - 'markers' 객체에는 오직 뮤트 줄 번호에 'x', 개방현에 'o'만 표기하세요 (프렛 번호 숫자 표기 절대 금지).\n\n"
        "반드시 아래 키 순서를 지키는 순수 JSON 객체 하나만 출력하세요 (모든 서술은 한국어로 작성):\n"
        "{\n"
        "  \"headstock_side\": \"left 또는 right (이미지에서 헤드스톡/1프렛이 있는 방향)\",\n"
        "  \"inlay_fret_check\": \"너트 또는 3,5,7,9,12프렛 인레이 점 기준 현재 손이 위치한 프렛 범위 서술\",\n"
        "  \"finger_observations\": \"1(검지/빨강), 2(중지/초록), 3(약지/파랑), 4(새끼/보라)가 지판 위쪽(6번줄)~아래쪽(1번줄) 중 각각 몇 번 줄, 몇 프렛을 누르고 있는지 보이는 그대로 서술\",\n"
        "  \"guitar_box\": [ymin, xmin, ymax, xmax],\n"
        "  \"base_fret\": 시작프렛정수,\n"
        "  \"barre\": {\"fret\": 바레프렛정수, \"start_string\": 시작줄번호, \"end_string\": 끝줄번호, \"finger\": 1} 또는 null,\n"
        "  \"dots\": [{\"string\": 줄번호1에서6, \"fret\": 프렛번호, \"finger\": 손가락번호1에서4}],\n"
        "  \"markers\": {\"6\": \"x\"},\n"
        "  \"fingering_summary\": \"프렛 위치 및 각 손가락(검지, 중지, 약지, 새끼)이 짚는 줄/프렛 핵심 요약 1~2문장\",\n"
        "  \"chord\": \"위 시각적 손가락 위치로 만들어지는 최종 코드명 (예: C, Am, Dm7, Gm7, E7(#9), F#m7 등)\"\n"
        "}"
    )


def run_core_analysis(full_frame: Image.Image, wav_bytes: bytes, mode: str):
    hand_info = guitar_detector.analyze_and_crop_guitar_hand(full_frame)
    if hand_info.get("only_picking_hand"):
        return {
            "detected_chord": "지판(왼손) 화면 필요",
            "clean_md": (
                "1. **상태 안내**: **코드를 잡는 지판(왼손) 화면이 필요합니다.**\n"
                "2. **원인 분석**: 현재 화면은 기타 사운드홀/바디 위에서 줄을 튕기거나 스트로크하는 **오른손(피킹 손)**만 클로즈업되어 있습니다.\n"
                "3. **해결 방법**: 코드를 누르고 있는 **기타 넥(지판)과 왼손이 보이는 장면**에서 일시정지 후 다시 분석해 주세요."
            ),
            "chord_svg": chord_visualizer.generate_chord_svg("C", prefer_fret=1),
            "clean_zoom_img": full_frame,
            "annotated_zoom_img": full_frame,
            "annotated_full_frame": full_frame,
            "wav_bytes": wav_bytes,
            "used_model": "picking-hand-guard",
        }

    clean_zoom_img = hand_info["clean_zoom_pil"]
    annotated_zoom_img = hand_info["annotated_zoom_pil"]
    cv_box_1000 = hand_info["box_1000"]

    dsp_info = audio_dsp.analyze_audio_wav_bytes(wav_bytes) if wav_bytes else {}

    hand_section = f"\n{hand_info['hand_summary_text']}\n" if hand_info.get("hand_summary_text") else ""
    if dsp_info.get("summary_text"):
        dsp_section = f"\n[±0.75초 오디오 참고 주파수]\n{dsp_info['summary_text']}\n"
    elif mode == "image":
        dsp_section = "\n[오디오 정보: 단일 사진 분석 모드로 오디오 없음 (지판 시각 관찰 최우선 판별)]\n"
    else:
        dsp_section = "\n[오디오 정보: 무음 또는 오디오 트랙 없음 (지판 시각 관찰 최우선 판별)]\n"

    prompt = build_vision_first_prompt(hand_section, dsp_section)
    raw_response, used_model = call_gemini_vision_first_fast(
        DEFAULT_API_KEYS,
        full_frame,
        clean_zoom_img,
        annotated_zoom_img,
        prompt,
    )

    clean_md, detected_chord, fingering_dict, prefer_fret = chord_visualizer.parse_ai_response(raw_response)
    annotated_full_frame = guitar_detector.draw_detection_box(full_frame, cv_box_1000)
    chord_svg = chord_visualizer.generate_chord_svg(
        detected_chord,
        fingering_dict=fingering_dict,
        prefer_fret=prefer_fret,
    )

    return {
        "detected_chord": detected_chord,
        "clean_md": clean_md,
        "chord_svg": chord_svg,
        "clean_zoom_img": clean_zoom_img,
        "annotated_zoom_img": annotated_zoom_img,
        "annotated_full_frame": annotated_full_frame,
        "wav_bytes": wav_bytes,
        "used_model": used_model,
    }


# ==============================================================================
# 4. 메인 UI 레이아웃
# ==============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 동영상 또는 사진 업로드")

    uploaded_file = st.file_uploader(
        "기타 연주 동영상 또는 지판 사진을 업로드해 주세요",
        type=["mp4", "mov", "avi", "webm", "mkv", "png", "jpg", "jpeg", "webp", "bmp"],
        key="media_file_uploader",
    )

    mode = None
    uploaded_image = None
    temp_video_path = None
    local_video_dur = 0.0

    if uploaded_file is not None:
        f_ext = os.path.splitext(uploaded_file.name.lower())[1]
        if f_ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            mode = "image"
            try:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.image(uploaded_image, caption=f"업로드된 사진: {uploaded_file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"사진 불러오기 오류: {e}")
        elif f_ext in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
            mode = "video"
            st.video(uploaded_file)

            file_sig = (uploaded_file.name, uploaded_file.size)
            cached_meta = st.session_state.get("cached_upload_meta")
            if cached_meta and cached_meta.get("sig") == file_sig and os.path.exists(cached_meta.get("path", "")):
                temp_video_path = cached_meta["path"]
                local_video_dur = cached_meta["dur"]
            else:
                temp_dir = tempfile.gettempdir()
                safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", uploaded_file.name)
                temp_video_path = os.path.join(temp_dir, f"guitar_up_{safe_name}")
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                cap = cv2.VideoCapture(temp_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                local_video_dur = total_frames / fps if fps > 0 else 300.0
                cap.release()

                st.session_state["cached_upload_meta"] = {
                    "sig": file_sig,
                    "path": temp_video_path,
                    "dur": local_video_dur,
                }
    else:
        st.info("동영상 파일(MP4, MOV 등) 또는 지판 사진(PNG, JPG 등)을 업로드해 주세요.")

    st.markdown("---")
    st.subheader("2. 분석 시점 및 실행")

    manual_timestamp = 0.0
    if mode == "image":
        st.info("단일 사진 모드: 사진 속 지판 프렛과 손가락 운지 관절을 시각적으로 정밀 분석합니다.")
    elif mode == "video":
        components.html(
            """
            <script>
            (function() {
              try {
                const doc = window.parent.document;
                function syncVideoTime() {
                  const videos = doc.querySelectorAll('video');
                  videos.forEach(v => {
                    if (v.dataset.pauseSyncBound) return;
                    v.dataset.pauseSyncBound = "true";

                    const onEvent = () => {
                      const t = parseFloat((v.currentTime || 0).toFixed(2));
                      const input = doc.querySelector('.st-key-manual_time_input input');
                      if (input) {
                        const cur = parseFloat(input.value || 0);
                        if (Math.abs(cur - t) >= 0.08) {
                          input.focus();
                          const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                          setter.call(input, t.toFixed(2));
                          input.dispatchEvent(new Event('input', { bubbles: true }));
                          input.dispatchEvent(new Event('change', { bubbles: true }));
                          input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
                          input.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }));
                          input.blur();
                        }
                      }
                    };

                    v.addEventListener('pause', onEvent);
                    v.addEventListener('seeked', onEvent);
                  });
                }

                setInterval(syncVideoTime, 400);
                syncVideoTime();
              } catch(e) {}
            })();
            </script>
            """,
            height=0,
            width=0,
        )

        init_val = float(min(st.session_state.get("target_time", 0.0), local_video_dur))
        manual_timestamp = st.number_input(
            f"분석할 동영상 재생 시간 (초) [총 {local_video_dur:.1f}초]",
            min_value=0.0,
            max_value=float(max(1.0, local_video_dur)),
            value=init_val,
            step=0.1,
            format="%.2f",
            key="manual_time_input",
        )
        st.session_state["target_time"] = manual_timestamp

        preview_img = extract_full_frame_fast(temp_video_path, manual_timestamp)
        if preview_img:
            st.image(
                preview_img,
                caption=f"분석될 실제 프레임 미리보기 ({manual_timestamp:.2f}초 시점)",
                use_container_width=True,
            )
        st.caption(f"선택한 시점({manual_timestamp:.2f}초) 기준 ±0.75초 오디오(FFT)와 프레임을 정밀 분석합니다.")
    else:
        st.caption("파일을 업로드하면 분석 옵션이 활성화됩니다.")

    anal_submitted = st.button("코드 분석 시작", type="primary", use_container_width=True)

with col2:
    st.subheader("3. 분석 결과")

    if anal_submitted:
        if mode is None:
            st.warning("먼저 동영상이나 사진 파일을 업로드해 주세요.")
            st.stop()

        with st.spinner("MediaPipe 관절 추적 및 멀티모달 코드 분석 중..."):
            full_frame = None
            wav_bytes = b""

            if mode == "image":
                full_frame = uploaded_image
            elif mode == "video":
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_frame = executor.submit(extract_full_frame_fast, temp_video_path, manual_timestamp)
                    fut_audio = executor.submit(extract_audio_clip_fast, temp_video_path, manual_timestamp)
                    full_frame = fut_frame.result()
                    wav_bytes = fut_audio.result()

            if full_frame is None:
                st.error("해당 시간대의 동영상 프레임 또는 사진을 불러오지 못했습니다.")
                st.stop()

            try:
                core_res = run_core_analysis(full_frame, wav_bytes, mode)
            except Exception as e:
                st.error(f"분석 오류: {e}")
                st.stop()

            st.session_state["analysis_result"] = {
                "timestamp": manual_timestamp if mode == "video" else None,
                "detected_chord": core_res["detected_chord"],
                "clean_md": core_res["clean_md"],
                "chord_svg": core_res["chord_svg"],
                "clean_zoom_img": core_res["clean_zoom_img"],
                "annotated_zoom_img": core_res["annotated_zoom_img"],
                "annotated_full_frame": core_res["annotated_full_frame"],
                "wav_bytes": core_res["wav_bytes"],
                "mode": mode,
            }

    res = st.session_state.get("analysis_result")
    if res is not None:
        ts_val = res.get("timestamp")
        ts_caption = f" ({ts_val:.2f}초)" if ts_val is not None else " (사진 분석)"
        img_col, diag_col = st.columns([1.15, 0.85])

        with img_col:
            tab_clean, tab_skeleton, tab_full = st.tabs([
                "지판 클로즈업",
                "손가락 관절 추적 (1~4)",
                "전체 화면",
            ])
            with tab_clean:
                st.image(
                    res["clean_zoom_img"],
                    caption=f"포착된 지판 및 운지 클로즈업{ts_caption}",
                    use_container_width=True,
                )
            with tab_skeleton:
                st.image(
                    res["annotated_zoom_img"],
                    caption="손가락 관절 추적 (1:검지, 2:중지, 3:약지, 4:새끼)",
                    use_container_width=True,
                )
            with tab_full:
                st.image(
                    res["annotated_full_frame"],
                    caption="주황색 박스: 자동 인식된 기타 지판 및 운지 손 영역",
                    use_container_width=True,
                )

            if res.get("wav_bytes"):
                st.audio(res["wav_bytes"], format="audio/wav")
                if ts_val is not None:
                    st.caption(f"추출된 ±0.75초 오디오 구간 ({max(0.0, ts_val-0.75):.2f}s ~ {ts_val+0.75:.2f}s)")
                else:
                    st.caption("추출된 오디오 클립")

        with diag_col:
            st.markdown(f"#### [{res['detected_chord']}] 운지 가이드")
            st.markdown(f'<div class="chord-card">{res["chord_svg"]}</div>', unsafe_allow_html=True)

        st.markdown("### 분석 리포트")
        st.markdown(res["clean_md"])
    else:
        st.info("동영상 또는 지판 사진을 업로드하고 '코드 분석 시작' 버튼을 누르면 정밀 분석 결과가 표시됩니다.")
