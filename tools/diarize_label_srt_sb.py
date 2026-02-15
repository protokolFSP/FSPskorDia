"""
Token-free diarization (no gated HF models):
- VAD: webrtcvad
- Speaker embeddings: SpeechBrain ECAPA (public model, no token)
- Clustering: Agglomerative (cosine)
- Map diarization -> SRT lines by overlap
- Guess assistant speaker by greeting + useful-question ratio
- Optionally export assistant questions

Usage:
  python tools/diarize_label_srt_sb.py \
    --audio Sample.m4a \
    --srt Sample.srt \
    --out-dir transcripts_labeled \
    --max-seconds 1200 \
    --num-speakers 2 \
    --extract-questions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import srt
import webrtcvad

try:
    import torch
    import torchaudio
except Exception as e:
    raise SystemExit("torch/torchaudio missing. Install requirements + torch cpu wheels.") from e

from sklearn.cluster import AgglomerativeClustering

try:
    from speechbrain.inference.speaker import EncoderClassifier
except Exception as e:
    raise SystemExit("speechbrain missing. pip install -r requirements.txt") from e


QUESTION_HINT_RE = re.compile(
    r"^\s*(wie|was|wann|wo|welche|wieviel|haben|sind|können|koennen|dürfen|duerfen|"
    r"darf|würden|wuerden|bitte)\b|[?]\s*$",
    re.IGNORECASE,
)
FILLER_RE = re.compile(
    r"\b(wie\s+bitte|nochmal\s+bitte|bitte\s+wiederholen|entschuldigung|"
    r"verstehen\s+sie|haben\s+sie\s+mich\s+verstanden)\b",
    re.IGNORECASE,
)
GREET_RE = re.compile(
    r"\b(guten\s+tag|mein\s+name\s+ist|aufnahmegespr(ä|ae)ch|anamnese|station|"
    r"ich\s+w(ü|ue)rde\s+gerne|darf\s+ich|ich\s+m(ö|oe)chte)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpeechSeg:
    start: float
    end: float


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def _which(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise SystemExit(f"{bin_name} not found in PATH.")
    return p


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def to_mono_wav(audio_in: Path, wav_out: Path, max_seconds: Optional[float]) -> None:
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_in)]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_out)]
    _run(cmd)


def read_wav_mono_16k(path: Path) -> Tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise SystemExit("WAV must be mono 16-bit 16k. Use ffmpeg conversion.")
        pcm = wf.readframes(wf.getnframes())
        return pcm, wf.getframerate()


def frame_generator(pcm: bytes, sample_rate: int, frame_ms: int) -> Iterable[Tuple[bytes, float]]:
    bytes_per_sample = 2
    frame_len = int(sample_rate * (frame_ms / 1000.0)) * bytes_per_sample
    offset = 0
    t = 0.0
    step_s = frame_ms / 1000.0
    while offset + frame_len <= len(pcm):
        yield pcm[offset : offset + frame_len], t
        offset += frame_len
        t += step_s


def vad_segments(
    pcm: bytes,
    sample_rate: int,
    vad_mode: int,
    frame_ms: int,
    padding_ms: int,
    min_seg_ms: int,
    max_seg_s: float,
) -> List[SpeechSeg]:
    vad = webrtcvad.Vad(vad_mode)
    frames = list(frame_generator(pcm, sample_rate, frame_ms))
    if not frames:
        return []

    padding_frames = max(1, int(padding_ms / frame_ms))
    min_frames = max(1, int(min_seg_ms / frame_ms))

    triggered = False
    ring: List[Tuple[bytes, float, bool]] = []
    seg_start: Optional[float] = None
    segments: List[SpeechSeg] = []

    for frame, t in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        ring.append((frame, t, is_speech))
        if len(ring) > padding_frames:
            ring.pop(0)

        num_voiced = sum(1 for *_x, s in ring if s)
        num_unvoiced = len(ring) - num_voiced

        if not triggered:
            if num_voiced > 0.8 * len(ring):
                triggered = True
                seg_start = ring[0][1]
        else:
            if num_unvoiced > 0.8 * len(ring):
                seg_end = t + (frame_ms / 1000.0)
                if seg_start is not None and (seg_end - seg_start) * 1000 >= min_seg_ms:
                    segments.append(SpeechSeg(seg_start, seg_end))
                triggered = False
                seg_start = None

    if triggered and seg_start is not None:
        seg_end = frames[-1][1] + (frame_ms / 1000.0)
        if (seg_end - seg_start) * 1000 >= min_seg_ms:
            segments.append(SpeechSeg(seg_start, seg_end))

    # merge tiny gaps
    merged: List[SpeechSeg] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start - prev.end <= 0.25:
            merged[-1] = SpeechSeg(prev.start, max(prev.end, seg.end))
        else:
            merged.append(seg)

    # split too-long segments
    out: List[SpeechSeg] = []
    for seg in merged:
        dur = seg.end - seg.start
        if dur <= max_seg_s:
            out.append(seg)
            continue
        n = int(math.ceil(dur / max_seg_s))
        for i in range(n):
            s = seg.start + i * max_seg_s
            e = min(seg.end, s + max_seg_s)
            if (e - s) * 1000 >= min_seg_ms:
                out.append(SpeechSeg(s, e))

    return out


def load_waveform_16k_mono(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != 16000:
        raise SystemExit("Expected 16k wav. Convert with ffmpeg.")
    if wav.shape[0] != 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    return wav


def extract_embeddings(
    wav_path: Path,
    segs: List[SpeechSeg],
    device: str = "cpu",
) -> np.ndarray:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    wav = load_waveform_16k_mono(wav_path)  # [1, T]
    sr = 16000
    embs: List[np.ndarray] = []

    for i, seg in enumerate(segs, 1):
        s = int(seg.start * sr)
        e = int(seg.end * sr)
        chunk = wav[:, s:e]
        if chunk.numel() < sr * 0.3:
            continue
        with torch.inference_mode():
            emb = classifier.encode_batch(chunk)  # [1, 1, D] or [1, D]
        emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        if emb.ndim != 1:
            emb = emb.reshape(-1)
        embs.append(emb)

        if i % 10 == 0:
            print(f"embedding {i}/{len(segs)}")

    if not embs:
        return np.zeros((0, 192), dtype=np.float32)
    X = np.stack(embs, axis=0)
    # L2 normalize for cosine
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X


def cluster_embeddings(X: np.ndarray, num_speakers: int) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([], dtype=int)
    k = max(1, min(num_speakers, X.shape[0]))
    try:
        cl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        cl = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return cl.fit_predict(X)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def turns_from_segments(segs: List[SpeechSeg], labels: np.ndarray) -> List[Turn]:
    if len(segs) != len(labels):
        # we may have dropped some short chunks during embedding; align by min length
        n = min(len(segs), len(labels))
        segs = segs[:n]
        labels = labels[:n]

    # order speakers by total duration -> SPEAKER_00 is the most talkative
    dur_by_lbl: Dict[int, float] = {}
    for seg, lbl in zip(segs, labels):
        dur_by_lbl[lbl] = dur_by_lbl.get(lbl, 0.0) + (seg.end - seg.start)
    lbl_sorted = sorted(dur_by_lbl.items(), key=lambda x: (-x[1], x[0]))
    lbl_map = {lbl: f"SPEAKER_{i:02d}" for i, (lbl, _d) in enumerate(lbl_sorted)}

    turns: List[Turn] = []
    for seg, lbl in zip(segs, labels):
        turns.append(Turn(seg.start, seg.end, lbl_map.get(int(lbl), "SPEAKER_99")))

    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def assign_speaker_to_sub(
    sub_start: float,
    sub_end: float,
    turns: List[Turn],
    min_overlap_ratio: float,
) -> str:
    dur = max(0.01, sub_end - sub_start)
    best_spk = "UNKNOWN"
    best_ov = 0.0
    for t in turns:
        if t.end <= sub_start:
            continue
        if t.start >= sub_end:
            break
        ov = overlap(sub_start, sub_end, t.start, t.end)
        if ov > best_ov:
            best_ov = ov
            best_spk = t.speaker
    if best_ov / dur < min_overlap_ratio:
        return "UNKNOWN"
    return best_spk


def is_useful_question(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if FILLER_RE.search(t):
        return False
    return bool(QUESTION_HINT_RE.search(t))


def guess_assistant_speaker(labeled: List[Tuple[str, srt.Subtitle]]) -> str:
    stats: Dict[str, Dict[str, int]] = {}
    for spk, sub in labeled:
        txt = sub.content.strip()
        if not txt:
            continue
        st = stats.setdefault(spk, {"total": 0, "greet": 0, "q": 0, "filler": 0})
        st["total"] += 1
        if GREET_RE.search(txt):
            st["greet"] += 1
        if is_useful_question(txt):
            st["q"] += 1
        if FILLER_RE.search(txt):
            st["filler"] += 1

    if not stats:
        return "UNKNOWN"

    def score(item: Tuple[str, Dict[str, int]]) -> Tuple[float, int, int]:
        spk, s = item
        total = max(1, s["total"])
        val = (6.0 * s["greet"]) + (2.5 * s["q"]) - (2.0 * s["filler"]) + (0.02 * total)
        return (val, s["q"], s["greet"])

    return max(stats.items(), key=score)[0]


def write_rttm(turns: List[Turn], file_id: str, out_rttm: Path) -> None:
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for t in turns:
        dur = max(0.0, t.end - t.start)
        lines.append(f"SPEAKER {file_id} 1 {t.start:.3f} {dur:.3f} <NA> <NA> {t.speaker} <NA> <NA>")
    out_rttm.write_text("\n".join(lines) + "\n", encoding="utf-8")


def trim_srt(subs: List[srt.Subtitle], max_seconds: Optional[float]) -> List[srt.Subtitle]:
    if not max_seconds or max_seconds <= 0:
        return subs
    out: List[srt.Subtitle] = []
    for sub in subs:
        st = sub.start.total_seconds()
        if st >= max_seconds:
            break
        en = min(sub.end.total_seconds(), max_seconds)
        out.append(
            srt.Subtitle(
                index=len(out) + 1,
                start=sub.start,
                end=srt.timedelta(seconds=en),
                content=sub.content,
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out-dir", default="transcripts_labeled")

    ap.add_argument("--max-seconds", type=float, default=1200.0)
    ap.add_argument("--num-speakers", type=int, default=2)

    ap.add_argument("--vad-mode", type=int, default=2)
    ap.add_argument("--frame-ms", type=int, default=30)
    ap.add_argument("--padding-ms", type=int, default=300)
    ap.add_argument("--min-seg-ms", type=int, default=800)
    ap.add_argument("--max-seg-s", type=float, default=12.0)

    ap.add_argument("--min-overlap-ratio", type=float, default=0.20)
    ap.add_argument("--extract-questions", action="store_true")
    args = ap.parse_args()

    _which("ffmpeg")

    audio_path = Path(args.audio)
    srt_path = Path(args.srt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_seconds = args.max_seconds if args.max_seconds and args.max_seconds > 0 else None

    wav_path = out_dir / "_wav" / f"{audio_path.stem}.wav"
    to_mono_wav(audio_path, wav_path, max_seconds)

    pcm, sr = read_wav_mono_16k(wav_path)
    segs = vad_segments(
        pcm=pcm,
        sample_rate=sr,
        vad_mode=args.vad_mode,
        frame_ms=args.frame_ms,
        padding_ms=args.padding_ms,
        min_seg_ms=args.min_seg_ms,
        max_seg_s=args.max_seg_s,
    )
    if not segs:
        raise SystemExit("No speech segments found by VAD.")

    print(f"VAD segments: {len(segs)}")

    X = extract_embeddings(wav_path, segs, device="cpu")
    if X.shape[0] == 0:
        raise SystemExit("No embeddings extracted (segments too short?).")

    labels = cluster_embeddings(X, args.num_speakers)
    turns = turns_from_segments(segs, labels)

    write_rttm(turns, audio_path.stem, out_dir / f"{audio_path.stem}.rttm")
    (out_dir / f"{audio_path.stem}.turns.json").write_text(
        json.dumps([t.__dict__ for t in turns], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subs_all = list(srt.parse(srt_path.read_text(encoding="utf-8", errors="ignore")))
    subs = trim_srt(subs_all, max_seconds)

    labeled: List[Tuple[str, srt.Subtitle]] = []
    for sub in subs:
        st = sub.start.total_seconds()
        en = sub.end.total_seconds()
        spk = assign_speaker_to_sub(st, en, turns, args.min_overlap_ratio)
        labeled.append((spk, sub))

    assistant_spk = guess_assistant_speaker(labeled)
    print(f"Assistant speaker guessed: {assistant_spk}")

    labeled_out: List[srt.Subtitle] = []
    assistant_only: List[srt.Subtitle] = []
    assistant_q: List[srt.Subtitle] = []
    q_counts: Dict[str, int] = {}

    for i, (spk, sub) in enumerate(labeled, 1):
        tagged = srt.Subtitle(i, sub.start, sub.end, f"{spk}: {sub.content}")
        labeled_out.append(tagged)
        if spk == assistant_spk:
            assistant_only.append(tagged)
            if args.extract_questions and is_useful_question(sub.content):
                assistant_q.append(tagged)
                q = re.sub(r"\s+", " ", sub.content.strip())
                q_counts[q] = q_counts.get(q, 0) + 1

    (out_dir / f"{audio_path.stem}.labeled.srt").write_text(srt.compose(labeled_out), encoding="utf-8")
    (out_dir / f"{audio_path.stem}.assistant_only.srt").write_text(srt.compose(assistant_only), encoding="utf-8")

    if args.extract_questions:
        (out_dir / f"{audio_path.stem}.assistant_questions.srt").write_text(srt.compose(assistant_q), encoding="utf-8")
        csv_path = out_dir / f"{audio_path.stem}.assistant_questions.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["count", "question"])
            for q, c in sorted(q_counts.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([c, q])

    print(f"Wrote: {out_dir / f'{audio_path.stem}.labeled.srt'}")
    print(f"Wrote: {out_dir / f'{audio_path.stem}.assistant_only.srt'}")
    if args.extract_questions:
        print(f"Wrote: {out_dir / f'{audio_path.stem}.assistant_questions.srt'}")
        print(f"Wrote: {out_dir / f'{audio_path.stem}.assistant_questions.csv'}")
    print(f"Wrote: {out_dir / f'{audio_path.stem}.rttm'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
