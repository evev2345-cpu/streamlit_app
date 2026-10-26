"""
일렉기타/어쿠스틱 기타 오디오 신호 처리(DSP) 기반 음정(Pitch Class) 및 화성(Chord) 분석 모듈
NumPy 벡터화 FFT, Subharmonic Summation(기본음 강화 및 배음 억제), 중앙 집중 윈도우(±0.25s / ±0.75s)를 결합하여
일렉기타 특유의 배음(오버드라이브/컴프레서)과 텐션 코드(7th, 9th, 13th, sus4, 파워코드 등)를 초고속(<2ms)으로 정밀 분석합니다.
"""

import io
import warnings
import numpy as np
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# 일렉/어쿠스틱 기타 주요 코드 인터벌 템플릿 (루트=0 기준 반음 간격)
CHORD_TEMPLATES = {
    '':        [0, 4, 7],             # Major
    'm':       [0, 3, 7],             # Minor
    '7':       [0, 4, 7, 10],         # Dominant 7
    'm7':      [0, 3, 7, 10],         # Minor 7
    'maj7':    [0, 4, 7, 11],         # Major 7
    '9':       [0, 4, 7, 10, 2],      # Dominant 9 (펑크/재즈 빈출)
    'm9':      [0, 3, 7, 10, 2],      # Minor 9 (네오소울/퓨전 빈출)
    'maj9':    [0, 4, 7, 11, 2],      # Major 9
    '7(b9)':   [0, 4, 7, 10, 1],      # 7b9
    '7(#9)':   [0, 4, 7, 10, 3],      # 7#9 (Hendrix chord)
    '7sus4':   [0, 5, 7, 10],         # 7sus4 / 9sus4 (펑크 빈출)
    'sus4':    [0, 5, 7],             # sus4
    '6':       [0, 4, 7, 9],          # Major 6
    'm6':      [0, 3, 7, 9],          # Minor 6
    'add9':    [0, 4, 7, 2],          # add9
    'm7b5':    [0, 3, 6, 10],         # Half-diminished (m7b5)
    'dim7':    [0, 3, 6, 9],          # Diminished 7
    '5':       [0, 7],                # Power chord (일렉기타 빈출)
}


def _build_precomputed_templates():
    """모듈 로드 시 12키 x 18개 코드 템플릿 행렬을 사전 계산하여 매 호출 시 연산 오버헤드를 제거합니다."""
    entries = []
    for root in range(12):
        for qual, intervals in CHORD_TEMPLATES.items():
            template = np.zeros(12, dtype=np.float64)
            for i, iv in enumerate(intervals):
                pc = (root + iv) % 12
                if i == 0:
                    template[pc] = 1.25
                elif iv in (3, 4, 10, 11):
                    template[pc] = 1.05
                elif iv in (1, 2, 9):
                    template[pc] = 0.90
                else:
                    template[pc] = 0.80
            norm_t = float(np.linalg.norm(template))
            non_chord_mask = template == 0
            entries.append((f"{NOTE_NAMES[root]}{qual}", root, template / (norm_t + 1e-8), non_chord_mask))
    return entries


_PRECOMPUTED_TEMPLATES = _build_precomputed_templates()


def _compute_chroma(signal: np.ndarray, sr: int, gaussian_sigma: float = 0.45):
    """주어진 오디오 신호에 가우시안-해닝 윈도우 및 배음 보정을 벡터화 적용해 12음계 크로마와 저음 크로마를 반환"""
    n = len(signal)
    if n < 256:
        return np.zeros(12), np.zeros(12)

    t_norm = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    gaussian_focus = np.exp(-0.5 * ((t_norm / gaussian_sigma) ** 2))
    windowed = signal * (np.hanning(n).astype(np.float32) * gaussian_focus)

    n_fft = max(65536, 1 << (n - 1).bit_length())
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    df = sr / n_fft

    clean_spec = spectrum.copy()
    min_idx = int(78.0 / df)
    max_idx = min(len(clean_spec) - 1, int(800.0 / df))
    src_indices = np.arange(min_idx, max_idx)

    # 1) 기본음(f0) 보강 (벡터 연산)
    dbl_indices = src_indices * 2
    valid_dbl = dbl_indices < len(spectrum)
    clean_spec[src_indices[valid_dbl]] += 0.20 * spectrum[dbl_indices[valid_dbl]]

    # 2) 상위 배음(2~5배음) 억제 (벡터 연산)
    for h, weight in ((2, 0.35), (3, 0.30), (4, 0.20), (5, 0.15)):
        h_indices = src_indices * h
        valid_h = h_indices < len(clean_spec)
        target_idx = h_indices[valid_h]
        clean_spec[target_idx] = np.maximum(0.0, clean_spec[target_idx] - weight * spectrum[src_indices[valid_h]])

    # 3) 유효 주파수 대역(78Hz ~ 1350Hz) 벡터 크로마 집계
    valid_mask = (freqs >= 78.0) & (freqs <= 1350.0) & (clean_spec > 0)
    v_freqs = freqs[valid_mask]
    v_mags = clean_spec[valid_mask]

    chroma = np.zeros(12, dtype=np.float64)
    bass_chroma = np.zeros(12, dtype=np.float64)

    if len(v_freqs) > 0:
        midis = 69.0 + 12.0 * np.log2(v_freqs / 440.0)
        nearest_midis = np.rint(midis).astype(np.int32)
        in_tune_mask = np.abs(midis - nearest_midis) <= 0.32

        v_freqs = v_freqs[in_tune_mask]
        v_mags = v_mags[in_tune_mask]
        pcs = nearest_midis[in_tune_mask] % 12

        np.add.at(chroma, pcs, v_mags ** 1.5)

        bass_mask = (v_freqs >= 78.0) & (v_freqs <= 295.0)
        if np.any(bass_mask):
            bass_weights = (v_mags[bass_mask] ** 1.65) * (220.0 / v_freqs[bass_mask])
            np.add.at(bass_chroma, pcs[bass_mask], bass_weights)

    max_c = np.max(chroma)
    if max_c > 0:
        chroma /= max_c
    max_b = np.max(bass_chroma)
    if max_b > 0:
        bass_chroma /= max_b

    return chroma, bass_chroma


def analyze_audio_wav_bytes(wav_bytes: bytes) -> dict:
    """
    1.5초(±0.75초) WAV 바이트 데이터를 입력받아:
    - 정중앙(±0.25s) 순간 음정과 전체(±0.75s) 지속 음정을 결합 분석
    - 가장 강하게 울리는 구성음(Pitch Classes)
    - 가장 낮은 저음 현 피크(추정 근음/베이스음)
    - DSP 화성학 템플릿 매칭 상위 코드 후보 5개
    를 계산하여 반환합니다.
    """
    if not wav_bytes:
        return {}

    try:
        sr, data = wavfile.read(io.BytesIO(wav_bytes))
    except Exception:
        return {}

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val < 1e-5:
        return {}
    data /= max_val

    chroma_instant, bass_instant = _compute_chroma(data, sr, gaussian_sigma=0.28)
    chroma_full, bass_full = _compute_chroma(data, sr, gaussian_sigma=0.65)

    chroma = 0.65 * chroma_instant + 0.35 * chroma_full
    bass_chroma = 0.65 * bass_instant + 0.35 * bass_full

    max_c = np.max(chroma)
    if max_c <= 0:
        return {}

    chroma /= max_c
    max_b = np.max(bass_chroma)
    if max_b > 0:
        bass_chroma /= max_b

    sorted_pcs = np.argsort(chroma)[::-1]
    active_notes = [
        f"{NOTE_NAMES[pc]}({chroma[pc]*100:.0f}%)"
        for pc in sorted_pcs[:6]
        if chroma[pc] >= 0.22
    ]

    bass_pcs = np.argsort(bass_chroma)[::-1]
    estimated_bass = NOTE_NAMES[bass_pcs[0]] if bass_chroma[bass_pcs[0]] > 0.18 else NOTE_NAMES[sorted_pcs[0]]
    second_bass = NOTE_NAMES[bass_pcs[1]] if len(bass_pcs) > 1 and bass_chroma[bass_pcs[1]] > 0.35 else None

    norm_c = float(np.linalg.norm(chroma)) + 1e-8
    chroma_unit = chroma / norm_c

    chord_scores = []
    for chord_name, root, unit_template, non_chord_mask in _PRECOMPUTED_TEMPLATES:
        sim = float(np.dot(chroma_unit, unit_template))
        penalty = 0.16 * float(np.mean(chroma[non_chord_mask]))
        bass_bonus = 0.20 * float(bass_chroma[root])
        score = sim - penalty + bass_bonus
        chord_scores.append((score, chord_name))

    chord_scores.sort(key=lambda x: x[0], reverse=True)
    top_chords = [name for _, name in chord_scores[:5]]

    bass_str = f"{estimated_bass}" + (f" (차순위 저음: {second_bass})" if second_bass else "")

    return {
        "active_notes": active_notes,
        "estimated_bass": estimated_bass,
        "top_chords": top_chords,
        "summary_text": (
            f"- 오디오 FFT 감지 음정(참고용 보조 지표): {', '.join(active_notes)}\n"
            f"- 저음역 피크 음(베이스/근음 후보): {bass_str}\n"
            f"- 오디오 화성 템플릿 매칭 상위 후보: {', '.join(top_chords)}"
        ),
    }
