"""
기타 코드 지판 다이어그램 SVG 생성기 (Vision-First 구조화 데이터 & 전 코드 보이싱 지원)
사용자가 제공한 참고 이미지(가로 6줄, 세로 프렛, 손가락 번호 원 ①~④, 바레 캡슐 박스, 프렛 번호) 스타일을 재현합니다.
"""

import json
import re

NOTE_TO_SEMITONE = {
    'C': 0, 'B#': 0,
    'C#': 1, 'DB': 1,
    'D': 2,
    'D#': 3, 'EB': 3,
    'E': 4, 'FB': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'GB': 6,
    'G': 7,
    'G#': 8, 'AB': 8,
    'A': 9,
    'A#': 10, 'BB': 10,
    'B': 11, 'CB': 11,
}

OPEN_CHORD_DB = {
    'C': {
        'base_fret': 1,
        'dots': [
            {'string': 2, 'fret': 1, 'finger': 1},
            {'string': 4, 'fret': 2, 'finger': 2},
            {'string': 5, 'fret': 3, 'finger': 3},
        ],
        'markers': {6: 'x'}
    },
    'C7': {
        'base_fret': 1,
        'dots': [
            {'string': 2, 'fret': 1, 'finger': 1},
            {'string': 4, 'fret': 2, 'finger': 2},
            {'string': 5, 'fret': 3, 'finger': 3},
            {'string': 3, 'fret': 3, 'finger': 4},
        ],
        'markers': {6: 'x'}
    },
    'CMAJ7': {
        'base_fret': 1,
        'dots': [
            {'string': 4, 'fret': 2, 'finger': 2},
            {'string': 5, 'fret': 3, 'finger': 3},
        ],
        'markers': {6: 'x'}
    },
    'D': {
        'base_fret': 1,
        'dots': [
            {'string': 3, 'fret': 2, 'finger': 1},
            {'string': 1, 'fret': 2, 'finger': 2},
            {'string': 2, 'fret': 3, 'finger': 3},
        ],
        'markers': {6: 'x', 5: 'x'}
    },
    'DM': {
        'base_fret': 1,
        'dots': [
            {'string': 1, 'fret': 1, 'finger': 1},
            {'string': 3, 'fret': 2, 'finger': 2},
            {'string': 2, 'fret': 3, 'finger': 4},
        ],
        'markers': {6: 'x', 5: 'x'}
    },
    'D7': {
        'base_fret': 1,
        'dots': [
            {'string': 2, 'fret': 1, 'finger': 1},
            {'string': 3, 'fret': 2, 'finger': 2},
            {'string': 1, 'fret': 2, 'finger': 3},
        ],
        'markers': {6: 'x', 5: 'x'}
    },
    'DM7': {
        'base_fret': 1,
        'barre': {'fret': 1, 'start_string': 1, 'end_string': 2, 'finger': 1},
        'dots': [
            {'string': 3, 'fret': 2, 'finger': 2},
        ],
        'markers': {6: 'x', 5: 'x'}
    },
    'E': {
        'base_fret': 1,
        'dots': [
            {'string': 3, 'fret': 1, 'finger': 1},
            {'string': 5, 'fret': 2, 'finger': 2},
            {'string': 4, 'fret': 2, 'finger': 3},
        ],
        'markers': {}
    },
    'EM': {
        'base_fret': 1,
        'dots': [
            {'string': 5, 'fret': 2, 'finger': 2},
            {'string': 4, 'fret': 2, 'finger': 3},
        ],
        'markers': {}
    },
    'E7': {
        'base_fret': 1,
        'dots': [
            {'string': 3, 'fret': 1, 'finger': 1},
            {'string': 5, 'fret': 2, 'finger': 2},
        ],
        'markers': {}
    },
    'EM7': {
        'base_fret': 1,
        'dots': [
            {'string': 5, 'fret': 2, 'finger': 2},
        ],
        'markers': {}
    },
    'G': {
        'base_fret': 1,
        'dots': [
            {'string': 5, 'fret': 2, 'finger': 2},
            {'string': 6, 'fret': 3, 'finger': 3},
            {'string': 1, 'fret': 3, 'finger': 4},
        ],
        'markers': {}
    },
    'G7': {
        'base_fret': 1,
        'dots': [
            {'string': 1, 'fret': 1, 'finger': 1},
            {'string': 5, 'fret': 2, 'finger': 2},
            {'string': 6, 'fret': 3, 'finger': 3},
        ],
        'markers': {}
    },
    'A': {
        'base_fret': 1,
        'dots': [
            {'string': 4, 'fret': 2, 'finger': 2},
            {'string': 3, 'fret': 2, 'finger': 3},
            {'string': 2, 'fret': 2, 'finger': 4},
        ],
        'markers': {6: 'x'}
    },
    'AM': {
        'base_fret': 1,
        'dots': [
            {'string': 2, 'fret': 1, 'finger': 1},
            {'string': 4, 'fret': 2, 'finger': 2},
            {'string': 3, 'fret': 2, 'finger': 3},
        ],
        'markers': {6: 'x'}
    },
    'A7': {
        'base_fret': 1,
        'dots': [
            {'string': 4, 'fret': 2, 'finger': 2},
            {'string': 2, 'fret': 2, 'finger': 3},
        ],
        'markers': {6: 'x'}
    },
    'AM7': {
        'base_fret': 1,
        'dots': [
            {'string': 2, 'fret': 1, 'finger': 1},
            {'string': 4, 'fret': 2, 'finger': 2},
        ],
        'markers': {6: 'x'}
    },
    'B7': {
        'base_fret': 1,
        'dots': [
            {'string': 4, 'fret': 1, 'finger': 1},
            {'string': 5, 'fret': 2, 'finger': 2},
            {'string': 3, 'fret': 2, 'finger': 3},
            {'string': 1, 'fret': 2, 'finger': 4},
        ],
        'markers': {6: 'x'}
    },
}


def build_movable_voicing(root_semi, quality, prefer_high_fret=None):
    fret_6 = (root_semi - 4) % 12
    if fret_6 == 0:
        fret_6 = 12
    fret_5 = (root_semi - 9) % 12
    if fret_5 == 0:
        fret_5 = 12

    q = quality.lower().strip()

    if q in ['9', 'm9', 'maj9', '7b9', '7(b9)', '7#9', '7(#9)']:
        r_fret = fret_5
        if prefer_high_fret and abs(fret_5 + 12 - prefer_high_fret) < abs(fret_5 - prefer_high_fret):
            r_fret = fret_5 + 12
        base = max(1, r_fret - 1)
        if q == 'm9':
            return {
                'base_fret': base,
                'dots': [
                    {'string': 4, 'fret': r_fret - 1, 'finger': 1},
                    {'string': 5, 'fret': r_fret, 'finger': 2},
                    {'string': 3, 'fret': r_fret, 'finger': 3},
                    {'string': 2, 'fret': r_fret, 'finger': 4},
                ],
                'markers': {6: 'x', 1: 'x'}
            }
        elif q in ['7b9', '7(b9)']:
            return {
                'base_fret': base,
                'dots': [
                    {'string': 4, 'fret': r_fret - 1, 'finger': 1},
                    {'string': 2, 'fret': r_fret - 1, 'finger': 1},
                    {'string': 5, 'fret': r_fret, 'finger': 2},
                    {'string': 3, 'fret': r_fret, 'finger': 3},
                ],
                'markers': {6: 'x', 1: 'x'}
            }
        elif q in ['7#9', '7(#9)']:
            return {
                'base_fret': base,
                'dots': [
                    {'string': 4, 'fret': r_fret - 1, 'finger': 1},
                    {'string': 5, 'fret': r_fret, 'finger': 2},
                    {'string': 3, 'fret': r_fret, 'finger': 3},
                    {'string': 2, 'fret': r_fret + 1, 'finger': 4},
                ],
                'markers': {6: 'x', 1: 'x'}
            }
        else:
            return {
                'base_fret': base,
                'barre': {'fret': r_fret, 'start_string': 1, 'end_string': 3, 'finger': 3},
                'dots': [
                    {'string': 4, 'fret': r_fret - 1, 'finger': 1},
                    {'string': 5, 'fret': r_fret, 'finger': 2},
                ],
                'markers': {6: 'x'}
            }

    use_5th_string = False
    if prefer_high_fret is not None:
        dist_6 = min(abs(fret_6 - prefer_high_fret), abs((fret_6 % 12) - prefer_high_fret))
        dist_5 = min(abs(fret_5 - prefer_high_fret), abs((fret_5 % 12) - prefer_high_fret))
        use_5th_string = dist_5 < dist_6
    else:
        if fret_5 <= 7 and fret_6 > 7:
            use_5th_string = True

    if use_5th_string:
        bf = fret_5
        if q in ['m', 'min', 'minor', '-']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 5, 'finger': 1},
                'dots': [
                    {'string': 2, 'fret': bf + 1, 'finger': 2},
                    {'string': 4, 'fret': bf + 2, 'finger': 3},
                    {'string': 3, 'fret': bf + 2, 'finger': 4},
                ],
                'markers': {6: 'x'}
            }
        elif q in ['m7', 'min7', '-7']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 5, 'finger': 1},
                'dots': [
                    {'string': 2, 'fret': bf + 1, 'finger': 2},
                    {'string': 4, 'fret': bf + 2, 'finger': 3},
                ],
                'markers': {6: 'x'}
            }
        elif q in ['7', 'dom7']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 5, 'finger': 1},
                'dots': [
                    {'string': 4, 'fret': bf + 2, 'finger': 3},
                    {'string': 2, 'fret': bf + 2, 'finger': 4},
                ],
                'markers': {6: 'x'}
            }
        elif q in ['maj7', 'm7+', 'Δ7']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 5, 'finger': 1},
                'dots': [
                    {'string': 3, 'fret': bf + 1, 'finger': 2},
                    {'string': 4, 'fret': bf + 2, 'finger': 3},
                    {'string': 2, 'fret': bf + 2, 'finger': 4},
                ],
                'markers': {6: 'x'}
            }
        else:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 5, 'finger': 1},
                'dots': [
                    {'string': 4, 'fret': bf + 2, 'finger': 2},
                    {'string': 3, 'fret': bf + 2, 'finger': 3},
                    {'string': 2, 'fret': bf + 2, 'finger': 4},
                ],
                'markers': {6: 'x'}
            }
    else:
        bf = fret_6
        if q in ['m', 'min', 'minor', '-']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 6, 'finger': 1},
                'dots': [
                    {'string': 5, 'fret': bf + 2, 'finger': 3},
                    {'string': 4, 'fret': bf + 2, 'finger': 4},
                ],
                'markers': {}
            }
        elif q in ['m7', 'min7', '-7']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 6, 'finger': 1},
                'dots': [
                    {'string': 5, 'fret': bf + 2, 'finger': 3},
                ],
                'markers': {}
            }
        elif q in ['7', 'dom7']:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 6, 'finger': 1},
                'dots': [
                    {'string': 3, 'fret': bf + 1, 'finger': 2},
                    {'string': 5, 'fret': bf + 2, 'finger': 3},
                ],
                'markers': {}
            }
        elif q in ['maj7', 'Δ7']:
            return {
                'base_fret': bf,
                'dots': [
                    {'string': 6, 'fret': bf, 'finger': 1},
                    {'string': 4, 'fret': bf + 1, 'finger': 3},
                    {'string': 3, 'fret': bf + 1, 'finger': 4},
                    {'string': 2, 'fret': bf, 'finger': 2},
                ],
                'markers': {5: 'x', 1: 'x'}
            }
        else:
            return {
                'base_fret': bf,
                'barre': {'fret': bf, 'start_string': 1, 'end_string': 6, 'finger': 1},
                'dots': [
                    {'string': 3, 'fret': bf + 1, 'finger': 2},
                    {'string': 5, 'fret': bf + 2, 'finger': 3},
                    {'string': 4, 'fret': bf + 2, 'finger': 4},
                ],
                'markers': {}
            }


def resolve_chord_data(chord_name, prefer_fret=None):
    clean = str(chord_name or "C").strip()
    for p in ["**", "정확한 코드명:", "추정 코드명:", "코드명:", "코드:"]:
        clean = clean.replace(p, "").strip()
    clean_main = clean.split("/")[0].split()[0].strip() if clean else "C"
    upper_key = clean.upper().replace(" MAJOR", "").strip()

    if (prefer_fret is None or prefer_fret <= 3) and upper_key in OPEN_CHORD_DB:
        return OPEN_CHORD_DB[upper_key]

    m = re.match(r'^([A-G][#b♯♭]?)(.*)$', clean_main, re.IGNORECASE)
    if not m:
        return OPEN_CHORD_DB.get('C')

    root_str = m.group(1).upper().replace('♯', '#').replace('♭', 'B')
    quality_str = m.group(2).strip()
    if quality_str.lower() in ['major', 'maj']:
        quality_str = ''

    root_semi = NOTE_TO_SEMITONE.get(root_str, 0)
    return build_movable_voicing(root_semi, quality_str, prefer_high_fret=prefer_fret)


def _safe_int(val, default=1):
    try:
        return int(val)
    except Exception:
        return default


def generate_chord_svg(chord_name, fingering_dict=None, prefer_fret=None):
    """
    참고 이미지 스타일의 지판 다이어그램 SVG 생성
    """
    valid_custom = False
    if fingering_dict and isinstance(fingering_dict, dict):
        dots = fingering_dict.get('dots')
        barre = fingering_dict.get('barre')
        if (isinstance(dots, list) and len(dots) > 0) or (isinstance(barre, dict) and barre.get('fret')):
            data = fingering_dict
            valid_custom = True

    if not valid_custom:
        data = resolve_chord_data(chord_name, prefer_fret=prefer_fret)

    all_frets = []
    for d in data.get('dots') or []:
        if isinstance(d, dict):
            f_val = _safe_int(d.get('fret'), 0)
            if f_val > 0:
                all_frets.append(f_val)

    barre = data.get('barre')
    if isinstance(barre, dict) and _safe_int(barre.get('fret'), 0) > 0:
        all_frets.append(_safe_int(barre['fret'], 1))

    base_fret = _safe_int(data.get('base_fret'), 1)
    if all_frets:
        min_f = min(all_frets)
        max_f = max(all_frets)
        if max_f <= 4 and min_f >= 1:
            base_fret = 1
        else:
            base_fret = max(1, min_f)

    width = 300
    height = 200
    start_x = 48
    fret_width = 48
    num_frets = 4
    start_y = 38
    string_gap = 22

    grid_w = fret_width * num_frets
    grid_h = string_gap * 5

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="background-color: #F8FAFC; border-radius: 12px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; '
        f'box-shadow: 0 4px 12px rgba(0,0,0,0.12); border: 1.5px solid #CBD5E1;">'
    ]

    svg.append(
        f'<text x="{start_x}" y="24" font-size="17" font-weight="800" fill="#0F1720">{chord_name}</text>'
    )

    svg.append(
        f'<rect x="{start_x}" y="{start_y}" width="{grid_w}" height="{grid_h}" fill="#E6EEF4" rx="2" />'
    )

    for f_idx in range(num_frets):
        actual_fret = base_fret + f_idx
        if actual_fret in [3, 5, 7, 9, 12, 15]:
            mx = start_x + f_idx * fret_width + (fret_width / 2)
            my = start_y + 2.5 * string_gap + 4
            svg.append(
                f'<text x="{mx}" y="{my}" font-size="13" font-weight="bold" fill="#475569" text-anchor="middle">*</text>'
            )

    for i in range(num_frets + 1):
        x = start_x + i * fret_width
        if i == 0 and base_fret == 1:
            svg.append(f'<line x1="{x-2}" y1="{start_y}" x2="{x-2}" y2="{start_y + grid_h}" stroke="#0F1720" stroke-width="2" />')
            svg.append(f'<line x1="{x+1}" y1="{start_y}" x2="{x+1}" y2="{start_y + grid_h}" stroke="#0F1720" stroke-width="2.5" />')
        else:
            svg.append(f'<line x1="{x}" y1="{start_y}" x2="{x}" y2="{start_y + grid_h}" stroke="#64748B" stroke-width="1.3" />')

    for i in range(6):
        y = start_y + i * string_gap
        str_w = 0.9 + (i * 0.3)
        svg.append(f'<line x1="{start_x}" y1="{y}" x2="{start_x + grid_w}" y2="{y}" stroke="#334155" stroke-width="{str_w:.1f}" />')

    if base_fret > 1:
        for f in range(num_frets):
            fx = start_x + f * fret_width + (fret_width / 2)
            svg.append(
                f'<text x="{fx}" y="{start_y + grid_h + 18}" font-size="12" font-weight="700" fill="#1E293B" text-anchor="middle">{base_fret + f}</text>'
            )

    markers = data.get('markers') or {}
    if isinstance(markers, dict):
        for s_key, marker in markers.items():
            s_num = _safe_int(s_key, 0)
            if marker and 1 <= s_num <= 6:
                y = start_y + (s_num - 1) * string_gap
                svg.append(
                    f'<text x="{start_x - 14}" y="{y + 4}" font-size="13" font-weight="800" fill="#0F1720" text-anchor="middle">{marker}</text>'
                )

    barre_covered_strings = set()
    if isinstance(barre, dict) and _safe_int(barre.get('fret'), 0) > 0:
        b_fret_num = _safe_int(barre['fret'], base_fret)
        b_fret = b_fret_num - base_fret
        if 0 <= b_fret < num_frets:
            bx = start_x + b_fret * fret_width + (fret_width / 2)
            s_start = min(_safe_int(barre.get('start_string'), 1), _safe_int(barre.get('end_string'), 6))
            s_end = max(_safe_int(barre.get('start_string'), 1), _safe_int(barre.get('end_string'), 6))
            s_start = max(1, min(6, s_start))
            s_end = max(1, min(6, s_end))
            y_top = start_y + (s_start - 1) * string_gap
            y_bot = start_y + (s_end - 1) * string_gap
            b_h = (y_bot - y_top) + 22
            svg.append(
                f'<rect x="{bx - 11}" y="{y_top - 11}" width="22" height="{b_h}" rx="7" fill="none" stroke="#0F1720" stroke-width="2.2" />'
            )
            b_finger = _safe_int(barre.get('finger'), 1)
            for s in [s_start, s_end]:
                barre_covered_strings.add((s, b_fret_num))
                sy = start_y + (s - 1) * string_gap
                svg.append(f'<circle cx="{bx}" cy="{sy}" r="8.5" fill="#FFFFFF" stroke="#0F1720" stroke-width="1.8" />')
                svg.append(f'<text x="{bx}" y="{sy + 4}" font-size="11" font-weight="800" fill="#0F1720" text-anchor="middle">{b_finger}</text>')

    for d in data.get('dots') or []:
        if not isinstance(d, dict):
            continue
        s_num = _safe_int(d.get('string'), 0)
        f_num = _safe_int(d.get('fret'), 0)
        if (s_num, f_num) in barre_covered_strings:
            continue
        d_fret = f_num - base_fret
        if 0 <= d_fret < num_frets and 1 <= s_num <= 6:
            cx = start_x + d_fret * fret_width + (fret_width / 2)
            cy = start_y + (s_num - 1) * string_gap
            finger = _safe_int(d.get('finger'), 1)
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="9" fill="#FFFFFF" stroke="#0F1720" stroke-width="1.8" />')
            svg.append(f'<text x="{cx}" y="{cy + 4}" font-size="11" font-weight="800" fill="#0F1720" text-anchor="middle">{finger}</text>')

    svg.append('</svg>')
    return "".join(svg)


def parse_ai_response(text: str):
    """
    Vision-First JSON 응답 또는 마크다운+JSON 응답에서:
    1) 화면에 표시할 2줄 마크다운 리포트 (1. 정확한 코드명, 2. 지판 위치 및 운지 요약)
    2) 추출된 코드명
    3) 구조화된 운지 딕셔너리 (base_fret, barre, dots, markers)
    4) 시작 프렛 번호
    를 반환합니다.
    """
    fingering_dict = None
    prefer_fret = None

    # 1. 전체가 JSON이거나 코드 블록 안에 JSON이 있는 경우 파싱
    cleaned_raw = text.strip()
    json_str = None
    fence_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', cleaned_raw, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)
    elif cleaned_raw.startswith("{") and cleaned_raw.endswith("}"):
        json_str = cleaned_raw
    else:
        brace_start = cleaned_raw.find("{")
        brace_end = cleaned_raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            json_str = cleaned_raw[brace_start:brace_end + 1]

    if json_str:
        try:
            fingering_dict = json.loads(json_str)
        except Exception:
            fingering_dict = None

    if fingering_dict and isinstance(fingering_dict, dict):
        detected_chord = str(fingering_dict.get("chord") or "C").strip()
        summary = str(fingering_dict.get("fingering_summary") or "").strip()
        visual_obs = str(fingering_dict.get("finger_observations") or "").strip()
        prefer_fret = _safe_int(fingering_dict.get("base_fret"), None)

        if not summary and visual_obs:
            summary = visual_obs

        # 영어로 출력되었거나 내용이 부실한 경우 한국어 자동 구성
        is_english = any(w in summary.lower() for w in ["finger", "fret", "string", "barre", "plays", "chord", "index", "middle", "ring", "pinky"]) and not re.search(r'[가-힣]', summary)
        if is_english or not summary:
            finger_names = {1: "검지", 2: "중지", 3: "약지", 4: "새끼손가락"}
            parts = []
            barre = fingering_dict.get("barre")
            if isinstance(barre, dict) and barre.get("fret"):
                fn = finger_names.get(barre.get("finger", 1), "검지")
                parts.append(f"{fn}로 {barre['fret']}프렛 ({barre.get('start_string', 1)}~{barre.get('end_string', 6)}번 줄) 바레")
            dots = fingering_dict.get("dots")
            if isinstance(dots, list):
                for d in dots:
                    if isinstance(d, dict) and "string" in d and "fret" in d:
                        fn = finger_names.get(d.get("finger", 1), f"{d.get('finger', 1)}번 손가락")
                        parts.append(f"{fn}로 {d['string']}번 줄 {d['fret']}프렛")
            if parts:
                summary = f"{', '.join(parts)}을 운지하는 [{detected_chord}] 코드 폼입니다."
            else:
                summary = f"[{detected_chord}] 코드의 기본 운지 형태입니다."

        clean_markdown = (
            f"1. **정확한 코드명**: **{detected_chord}**\n"
            f"2. **지판 위치 및 운지 요약**: {summary}"
        )
        return clean_markdown, detected_chord, fingering_dict, prefer_fret

    # Fallback: 일반 마크다운 파싱
    clean_markdown = re.sub(r'```(?:json)?.*?```', '', cleaned_raw, flags=re.DOTALL).strip()
    detected_chord = "C"
    for line in clean_markdown.split("\n"):
        if "코드명" in line and ":" in line:
            part = line.split(":", 1)[1].replace("**", "").replace("*", "").strip()
            detected_chord = part.split("(")[0].split(",")[0].strip()
            if detected_chord:
                break

    fret_matches = re.findall(r'(\d{1,2})\s*(?:~|-|번)?\s*프렛', clean_markdown)
    if fret_matches:
        frets = [int(f) for f in fret_matches if 1 <= int(f) <= 22]
        if frets:
            prefer_fret = min(frets)

    return clean_markdown, detected_chord, None, prefer_fret
