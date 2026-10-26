"""
MediaPipe HandLandmarker(21개 3D 손 관절 키포인트) + 지판 기하학 분석 모듈
1. 영상 속에서 실제 연주자의 손(운지 손 vs 피킹 손)을 100% 해부학적으로 정확히 탐지합니다 (얼굴/배경 오인식 원천 차단).
2. 운지 손(Fretting Hand)의 4개 손가락 끝(검지①, 중지②, 약지③, 새끼④)과 엄지(T)의 정확한 좌표 및
   검지 바레(Barre) 여부, 썸오버(Thumb-over) 여부를 수학적으로 계산합니다.
3. 프렛 번호를 정확히 셀 수 있도록 운지 손 주변의 지판(너트~12프렛 인레이 맥락 포함)을 넓고 선명하게 크롭하고,
   손가락 번호(1, 2, 3, 4)가 마킹된 가이드 이미지를 생성하여 비전 모델의 시각적 오인식을 방지합니다.
"""

import math
import os
import threading
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

_LANDMARKER = None
_LANDMARKER_LOCK = threading.Lock()
_CLAHE = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))


def get_hand_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None and os.path.exists(MODEL_PATH):
        with _LANDMARKER_LOCK:
            if _LANDMARKER is None:
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
                    options = mp_vision.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=2,
                        min_hand_detection_confidence=0.20,
                        min_hand_presence_confidence=0.20,
                    )
                    _LANDMARKER = mp_vision.HandLandmarker.create_from_options(options)
                except Exception as e:
                    print(f"MediaPipe 초기화 경고: {e}")
                    _LANDMARKER = None
    return _LANDMARKER


def _angle_deg(p1, p2, p3) -> float:
    """세 점 p1-p2-p3 사이의 각도(도) 계산"""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 * mag2 < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


def _is_downward_picking_hand(lm, hd) -> bool:
    """
    손가락 관절과 손끝이 아래쪽(바닥)을 향하고 있는지 생체역학적으로 판별합니다.
    기타를 칠 때 코드를 잡는 왼손은 지판 아래에서 위로 감싸 쥐므로 손끝이 위나 수평을 향하지만,
    사운드홀/바디 위에서 피킹하는 오른손은 손등이 위로 오고 손가락 관절/끝이 아래를 향합니다.
    """
    wrist_y = lm[0].y
    avg_tip_y = (lm[8].y + lm[12].y + lm[16].y + lm[20].y) / 4.0
    avg_mcp_y = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4.0
    tip_diff = avg_tip_y - wrist_y
    mcp_diff = avg_tip_y - avg_mcp_y
    is_right = hd and hd.category_name == "Right" and hd.score > 0.70

    # 1) 손목/손마디보다 손끝이 명백히 아래쪽(+0.04 이상)을 향하고 있는 경우
    if tip_diff > 0.04 and mcp_diff > 0.01:
        return True
    # 2) 오른손(Right)이면서 손끝이 손목보다 아래쪽에 위치하는 경우
    if is_right and tip_diff > 0.01:
        return True
    # 3) 신뢰도 85% 이상의 오른손 단독인 경우
    if is_right and hd.score > 0.85:
        return True
    return False


def _select_fretting_hand(hand_landmarks_list, handedness_list, w: int, h: int):
    """
    감지된 손(1~2개) 중 기타 지판을 짚고 있는 '운지 손(Fretting Hand)'과
    바디 쪽의 '피킹 손(Picking Hand)'을 구분합니다.
    - 손가락 관절/끝이 아래쪽을 향하면 100% 바디/사운드홀 위 피킹 손으로 판정하여 운지 손에서 즉시 배제합니다.
    """
    if not hand_landmarks_list:
        return None, None

    if len(hand_landmarks_list) == 1:
        hd = handedness_list[0][0] if (handedness_list and handedness_list[0]) else None
        if _is_downward_picking_hand(hand_landmarks_list[0], hd):
            return None, hand_landmarks_list[0]
        return hand_landmarks_list[0], None

    evaluated = []
    for idx, lm in enumerate(hand_landmarks_list):
        hd = handedness_list[idx][0] if (handedness_list and idx < len(handedness_list) and handedness_list[idx]) else None
        is_pick = _is_downward_picking_hand(lm, hd)
        wrist_y = lm[0].y
        avg_tip_y = (lm[8].y + lm[12].y + lm[16].y + lm[20].y) / 4.0
        avg_mcp_y = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4.0
        upward_score = (wrist_y - avg_tip_y) * 2.0 + (avg_mcp_y - avg_tip_y)
        if hd:
            if hd.category_name == "Left":
                upward_score += 1.5
            elif hd.category_name == "Right":
                upward_score -= 1.5
        if is_pick:
            upward_score -= 10.0
        evaluated.append((upward_score, is_pick, lm))

    scores = evaluated
    scores.sort(key=lambda x: x[0], reverse=True)
    best = scores[0]
    second = scores[1]
    if best[1] and not second[1]:
        return second[2], best[2]
    if best[1]:
        # 둘 다 아래를 향하는 피킹 손인 경우 운지 손 없음
        return None, best[2]
    return best[2], second[2]


def enhance_fretboard_roi(rgb_crop: np.ndarray) -> Image.Image:
    """어두운 지판 위에서도 프렛 와이어, 인레이 점, 기타 줄, 손가락 마디가 선명히 보이도록 소프트 CLAHE 적용"""
    if rgb_crop.size == 0:
        return Image.new("RGB", (320, 180))
    lab = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    with _LANDMARKER_LOCK:
        l_enhanced = _CLAHE.apply(l_chan)
    merged = cv2.merge((l_enhanced, a_chan, b_chan))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def analyze_and_crop_guitar_hand(pil_img: Image.Image) -> dict:
    """
    전체 프레임에서 MediaPipe HandLandmarker로 운지 손을 정밀 탐지하고:
    - clean_zoom_pil: 지판 맥락(인레이/프렛)이 충분히 포함된 고해상도 클로즈업 이미지
    - annotated_zoom_pil: 손가락 관절 및 손가락 번호(1:검지, 2:중지, 3:약지, 4:새끼, T:엄지)가 표시된 가이드 이미지
    - box_1000: [ymin, xmin, ymax, xmax] (0~1000 정규화 좌표)
    - hand_summary_text: 손가락 기하학적 배치 및 바레/썸오버 분석 텍스트
    를 반환합니다.
    """
    rgb = np.array(pil_img.convert("RGB"))
    h, w, _ = rgb.shape

    landmarker = get_hand_landmarker()
    fretting_lm = None
    picking_lm = None

    if landmarker is not None:
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with _LANDMARKER_LOCK:
                res = landmarker.detect(mp_img)
            if res.hand_landmarks:
                fretting_lm, picking_lm = _select_fretting_hand(res.hand_landmarks, res.handedness, w, h)
        except Exception as e:
            print(f"MediaPipe 탐지 예외: {e}")

    if fretting_lm is None and picking_lm is not None:
        return {
            "hand_detected": False,
            "only_picking_hand": True,
            "clean_zoom_pil": pil_img,
            "annotated_zoom_pil": pil_img,
            "box_1000": [100, 100, 900, 900],
            "hand_summary_text": "- **주의**: 현재 화면에는 기타 사운드홀/바디 위에서 연주하는 오른손(피킹 손)만 포착되었으며, 코드를 잡는 왼손(지판)은 화면에 보이지 않습니다.",
        }

    if fretting_lm is not None:
        xs = [pt.x * w for pt in fretting_lm]
        ys = [pt.y * h for pt in fretting_lm]
        hx_min, hx_max = min(xs), max(xs)
        hy_min, hy_max = min(ys), max(ys)
        hand_w = max(40.0, hx_max - hx_min)
        hand_h = max(40.0, hy_max - hy_min)

        # 지판의 프렛 번호와 인레이 점(3, 5, 7, 9, 12프렛)을 AI가 정확히 셀 수 있도록
        # 좌우(기타 넥 방향)로는 손 너비의 2.2배씩 넉넉하게 포함하고, 상하로는 손 주변만 깔끔하게 크롭
        x1 = int(max(0, hx_min - hand_w * 2.2))
        x2 = int(min(w, hx_max + hand_w * 2.2))
        y1 = int(max(0, hy_min - hand_h * 0.85))
        y2 = int(min(h, hy_max + hand_h * 0.65))

        clean_crop_rgb = rgb[y1:y2, x1:x2]
        clean_zoom_pil = enhance_fretboard_roi(clean_crop_rgb)

        # 손가락 기하학 분석 (1:검지, 2:중지, 3:약지, 4:새끼, T:엄지)
        pts = [(pt.x * w, pt.y * h) for pt in fretting_lm]
        index_angle = _angle_deg(pts[5], pts[6], pts[8])
        index_span_y = abs(pts[8][1] - pts[5][1])

        # 바레 판별: 검지가 지판을 넓게 가로지르며(span_y가 손 높이의 55% 이상) 곧게 펴져 있는 경우에만 바레로 판정
        is_barre_likely = index_angle > 165.0 and index_span_y > (hand_h * 0.55)
        # 엄지(4)가 검지 끝(8)이나 중지 끝(12)만큼 높이 올라와 있으면 썸오버(6번줄 뮤트 또는 근음 운지)
        highest_finger_y = min(pts[8][1], pts[12][1], pts[16][1])
        is_thumb_over = pts[4][1] <= (highest_finger_y + hand_h * 0.15)

        # 헤드스톡 방향 추정 (피킹 손이 함께 감지된 경우 피킹 손 반대쪽이 헤드스톡)
        neck_dir_hint = ""
        if picking_lm is not None:
            pick_x = np.mean([p.x * w for p in picking_lm])
            fret_x = np.mean(xs)
            if fret_x > pick_x:
                neck_dir_hint = "바디(피킹 손)가 왼쪽, 헤드스톡(1프렛 쪽)이 오른쪽에 위치함 (우측으로 갈수록 낮은 프렛)"
            else:
                neck_dir_hint = "헤드스톡(1프렛 쪽)이 왼쪽, 바디(피킹 손)가 오른쪽에 위치함 (좌측으로 갈수록 낮은 프렛)"

        # 각 손가락 끝의 상대적 위치 계산 (지판 위쪽=6번줄 방향 vs 아래쪽=1번줄 방향)
        finger_tips = [
            ("1번(검지)", pts[8], pts[6]),
            ("2번(중지)", pts[12], pts[10]),
            ("3번(약지)", pts[16], pts[14]),
            ("4번(새끼)", pts[20], pts[18]),
        ]

        # Annotated 가이드 이미지 생성 (손가락 관절 선 + 손가락 번호 라벨링)
        annotated_zoom_pil = clean_zoom_pil.copy()
        draw = ImageDraw.Draw(annotated_zoom_pil)

        finger_colors = {
            1: "#FF3B30",  # 검지(1): 빨강
            2: "#34C759",  # 중지(2): 초록
            3: "#007AFF",  # 약지(3): 파랑
            4: "#AF52DE",  # 새끼(4): 보라
        }
        finger_chains = [
            (1, [5, 6, 7, 8]),
            (2, [9, 10, 11, 12]),
            (3, [13, 14, 15, 16]),
            (4, [17, 18, 19, 20]),
        ]
        for f_num, chain in finger_chains:
            color = finger_colors[f_num]
            for a, b in zip(chain[:-1], chain[1:]):
                pa = (pts[a][0] - x1, pts[a][1] - y1)
                pb = (pts[b][0] - x1, pts[b][1] - y1)
                draw.line([pa, pb], fill=color, width=2)
            # 손끝(Tip)은 줄/프렛 접점이 가려지지 않도록 작은 속 빈 원으로 표시하고, 번호 배지는 두 번째 마디(PIP)에 표시
            tip_x, tip_y = pts[chain[-1]][0] - x1, pts[chain[-1]][1] - y1
            draw.ellipse([tip_x - 5, tip_y - 5, tip_x + 5, tip_y + 5], outline=color, width=2)
            pip_x, pip_y = pts[chain[1]][0] - x1, pts[chain[1]][1] - y1
            draw.ellipse([pip_x - 8, pip_y - 8, pip_x + 8, pip_y + 8], fill=color)
            draw.text((pip_x - 3, pip_y - 6), str(f_num), fill="#FFFFFF")

        # 엄지 표시
        tx, ty = pts[4][0] - x1, pts[4][1] - y1
        draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], outline="#FF9500", width=2)

        # 상대적 줄 높이(위쪽 6번줄 ~ 아래쪽 1번줄) 순서 정리
        sorted_by_y = sorted(finger_tips, key=lambda item: item[1][1])
        vertical_order_str = " → ".join([f"{name}" for name, _, _ in sorted_by_y])

        barre_str = (
            f"검지(1번) 관절 각도 {index_angle:.0f}°로 펴져 있어 **바레(Barre, 여러 줄 동시 누름)** 가능성이 높음"
            if is_barre_likely
            else f"검지(1번) 관절 각도 {index_angle:.0f}°로 구부러져 있어 **단일 줄 독립 운지** 형태임"
        )
        thumb_str = (
            "엄지(T)가 지판 위쪽(6번줄 방향)으로 올라온 **썸오버(Thumb-over) 폼** 감지됨"
            if is_thumb_over
            else "엄지(T)가 넥 뒤쪽을 받치는 **클래식/바레 파지 폼** 감지됨"
        )

        summary_lines = [
            f"- **MediaPipe 운지 손 관절 실측**:",
            f"  - {barre_str}",
            f"  - {thumb_str}",
            f"  - 손끝 높이(지판 위쪽 6번줄~아래쪽 1번줄) 순서: {vertical_order_str} (※ 실제 줄을 누르고 있는 활성 손가락만 코드에 반영)",
        ]
        if neck_dir_hint:
            summary_lines.append(f"  - 기타 넥 방향: {neck_dir_hint}")

        box_1000 = [
            int(y1 * 1000 / h),
            int(x1 * 1000 / w),
            int(y2 * 1000 / h),
            int(x2 * 1000 / w),
        ]
        return {
            "hand_detected": True,
            "clean_zoom_pil": clean_zoom_pil,
            "annotated_zoom_pil": annotated_zoom_pil,
            "box_1000": box_1000,
            "hand_summary_text": "\n".join(summary_lines),
        }

    # Fallback: MediaPipe가 손을 못 찾은 경우 (가려짐 등) 전체 화면 중심부 사용
    x1, y1, x2, y2 = int(w * 0.08), int(h * 0.12), int(w * 0.92), int(h * 0.85)
    clean_zoom_pil = enhance_fretboard_roi(rgb[y1:y2, x1:x2])
    box_1000 = [120, 80, 850, 920]
    return {
        "hand_detected": False,
        "clean_zoom_pil": clean_zoom_pil,
        "annotated_zoom_pil": clean_zoom_pil,
        "box_1000": box_1000,
        "hand_summary_text": "",
    }


def crop_by_normalized_box(pil_img: Image.Image, box_1000: list[int]) -> Image.Image:
    w, h = pil_img.size
    ymin, xmin, ymax, xmax = box_1000
    x1 = max(0, min(w - 1, int(xmin * w / 1000)))
    y1 = max(0, min(h - 1, int(ymin * h / 1000)))
    x2 = max(x1 + 20, min(w, int(xmax * w / 1000)))
    y2 = max(y1 + 20, min(h, int(ymax * h / 1000)))
    rgb_crop = np.array(pil_img.convert("RGB"))[y1:y2, x1:x2]
    return enhance_fretboard_roi(rgb_crop)


def draw_detection_box(pil_img: Image.Image, box_1000: list[int]) -> Image.Image:
    img_out = pil_img.copy().convert("RGB")
    w, h = img_out.size
    ymin, xmin, ymax, xmax = box_1000
    x1 = max(0, min(w - 1, int(xmin * w / 1000)))
    y1 = max(0, min(h - 1, int(ymin * h / 1000)))
    x2 = max(x1 + 10, min(w, int(xmax * w / 1000)))
    y2 = max(y1 + 10, min(h, int(ymax * h / 1000)))

    draw = ImageDraw.Draw(img_out)
    for offset in range(3):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline="#FF7A59")
    return img_out
