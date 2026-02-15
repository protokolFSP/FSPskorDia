from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import srt
from pyannote.audio import Pipeline


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


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
    r"\b(guten\s+tag|mein\s+name\s+ist|ich\s+bin\s+(arzt|aerztin|assist|assistent|pflege)|"
    r"aufnahmegespr(ä|ae)ch|anamnese|station|ich\s+w(ü|ue)rde\s+gerne|"
    r"darf\s+ich|ich\s+m(ö|oe)chte)\b",
    re.IGNORECASE,
)


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _which(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise SystemExit(f"{bin_name} not found in PATH.")
    return p


def to_mono_wav(audio_in: Path, wav_out: Path, max_seconds: float | None) -> None:
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_in),
    ]
    if max_seconds is not None and max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_out)]
    _run(cmd)


def diarize(wav_path: Path, model_id: str, hf_token: str) -> List[Turn]:
    pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    diar = pipeline(str(wav_path))
    turns: List[Turn] = []
    for segment, _, speaker in diar.itertracks(yield_label=True):
        turns.append(Turn(float(segment.start), float(segment.end), str(speaker)))
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def write_rttm(turns: List[Turn], file_id: str, out_rttm: Path) -> None:
    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for t in turns:
        dur = max(0.0, t.end - t.start)
        lines.append(f"SPEAKER {file_id} 1 {t.start:.3f} {dur:.3f} <NA> <NA> {t.speaker} <NA> <NA>")
    out_rttm.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(sub_start: float, sub_end: float, turns: List[Turn], min_overlap_ratio: float) -> str:
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
    return bool(QUESTION_HINT_RE.search(t) and not FILLER_RE.search(t))


def guess_assistant_speaker(labeled_subs: List[Tuple[str, srt.Subtitle]]) -> str:
    # Score speaker by (greeting + useful questions - fillers)
    stats: Dict[str, Dict[str, int]] = {}
    for spk, sub in labeled_subs:
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
        val = (5.0 * s["greet"]) + (2.0 * s["q"]) - (2.0 * s["filler"]) + (0.05 * total)
        return (val, s["q"], s["greet"])

    return max(stats.items(), key=score)[0]


def trim_srt_to_max_seconds(subs: List[srt.Subtitle], max_seconds: float | None) -> List[srt.Subtitle]:
    if max_seconds is None or max_seconds <= 0:
        return subs
    out: List[srt.Subtitle] = []
    for sub in subs:
        st = sub.start.total_seconds()
        if st >= max_seconds:
            break
        en = min(sub.end.total_seconds(), max_seconds)
        out.append(
            srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=srt.timedelta(seconds=en),
                content=sub.content,
            )
        )
    # reindex
    return [srt.Subtitle(i + 1, x.start, x.end, x.content) for i, x in enumerate(out)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out-dir", default="transcripts_labeled")
    ap.add_argument("--max-seconds", type=float, default=1200.0)
    ap.add_argument("--min-overlap-ratio", type=float, default=0.20)
    ap.add_argument("--extract-questions", action="store_true")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    args = ap.parse_args()

    _which("ffmpeg")

    if not args.hf_token:
        raise SystemExit("HF token missing. Set HF_TOKEN env var or pass --hf-token.")

    audio_path = Path(args.audio)
    srt_path = Path(args.srt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_seconds = args.max_seconds if args.max_seconds and args.max_seconds > 0 else None

    wav_path = out_dir / "_wav" / f"{audio_path.stem}.wav"
    to_mono_wav(audio_path, wav_path, max_seconds)

    turns = diarize(wav_path, args.model, args.hf_token)
    write_rttm(turns, wav_path.stem, out_dir / f"{audio_path.stem}.rttm")

    subs_all = list(srt.parse(srt_path.read_text(encoding="utf-8", errors="ignore")))
    subs = trim_srt_to_max_seconds(subs_all, max_seconds)

    labeled: List[Tuple[str, srt.Subtitle]] = []
    for sub in subs:
        st = sub.start.total_seconds()
        en = sub.end.total_seconds()
        spk = assign_speaker(st, en, turns, args.min_overlap_ratio)
        labeled.append((spk, sub))

    assistant_spk = guess_assistant_speaker(labeled)

    labeled_out: List[srt.Subtitle] = []
    assistant_only_out: List[srt.Subtitle] = []
    assistant_questions: List[srt.Subtitle] = []
    q_counts: Dict[str, int] = {}

    for i, (spk, sub) in enumerate(labeled, 1):
        tagged = srt.Subtitle(
            index=i,
            start=sub.start,
            end=sub.end,
            content=f"{spk}: {sub.content}",
        )
        labeled_out.append(tagged)

        if spk == assistant_spk:
            assistant_only_out.append(tagged)
            if args.extract_questions and is_useful_question(sub.content):
                assistant_questions.append(tagged)
                norm_q = re.sub(r"\s+", " ", sub.content.strip())
                q_counts[norm_q] = q_counts.get(norm_q, 0) + 1

    (out_dir / f"{audio_path.stem}.labeled.srt").write_text(srt.compose(labeled_out), encoding="utf-8")
    (out_dir / f"{audio_path.stem}.assistant_only.srt").write_text(srt.compose(assistant_only_out), encoding="utf-8")

    if args.extract_questions:
        (out_dir / f"{audio_path.stem}.assistant_questions.srt").write_text(
            srt.compose(assistant_questions), encoding="utf-8"
        )
        csv_path = out_dir / f"{audio_path.stem}.assistant_questions.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["count", "question"])
            for q, c in sorted(q_counts.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([c, q])

    print(f"Assistant speaker guessed: {assistant_spk}")
    print(f"Wrote: {out_dir / f'{audio_path.stem}.labeled.srt'}")
    print(f"Wrote: {out_dir / f'{audio_path.stem}.assistant_only.srt'}")
    if args.extract_questions:
        print(f"Wrote: {out_dir / f'{audio_path.stem}.assistant_questions.srt'}")
        print(f"Wrote: {out_dir / f'{audio_path.stem}.assistant_questions.csv'}")
    print(f"Wrote: {out_dir / f'{audio_path.stem}.rttm'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
