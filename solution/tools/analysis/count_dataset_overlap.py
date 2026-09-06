"""Count how much the distributed LIBERO datasets overlap, as sets of instructions.

The datasets are handed out in three different LeRobot layouts, so the instruction
set is read from whichever meta file exists:

    meta/tasks.jsonl     LeRobot v2.1  (IPEC-COMMUNITY suites)
    meta/tasks.parquet   LeRobot v3.0  (lerobot/libero_plus)
    meta/episodes.jsonl  fallback      (Sylvest/libero_plus_lerobot, no tasks file)

Only meta/ is read: no video, no parquet shard of frames. A dataset that sits in a
tar can therefore be counted by extracting meta/ alone.

Instructions are matched against the bddl files under --bddl-root, which is the only
way to tell which suite a task came from: the datasets carry the instruction
sentence, and the SCENE name lives in the bddl file name. Two keys per bddl are
tried, because neither alone covers every suite:

    the (:language ...) line
    the file name with the SCENE prefix and the perturbation suffix removed

The LIBERO-plus variants rephrase the sentence ("pick up the black bowl ..." becomes
"pick the akita black bowl ..."), so the file name is what matches there.

Usage:
    python3 count_dataset_overlap.py \
        --meta libero_plus=/path/to/snapshot/meta \
        --meta libero_90=/path/to/snapshot/meta \
        --bddl-root work/LIBERO-plus/libero/libero/bddl_files
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

SCENE_RE = re.compile(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+")
LANGUAGE_RE = re.compile(r"\(:language\s+(.+?)\)")
# The eight shapes LIBERO-plus appends to a task bddl name when it perturbs it,
# read off `ls bddl_files/libero_10`: distractors, lighting, table texture, camera,
# rephrased instruction, the level/sample grid, and plain editor copies.
VARIANT_SUFFIX_RE = re.compile(
    r"(_add_\d+"
    r"| copy( \d+)?"
    r"|_language_\d+"
    r"|_level\d+_sample\d+"
    r"|_light_\d+"
    r"|_table_\d+"
    r"|_tb_\d+"
    r"|_view_\d+)$"
)


def normalize(instruction: str) -> str:
    """Fold the spelling differences that are not real task differences."""
    text = instruction.strip().lower().rstrip(".")
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text)


def read_instructions(meta_dir: Path) -> set[str]:
    """Return the set of instructions in one dataset, whatever its layout is."""
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        with tasks_jsonl.open() as f:
            return {normalize(json.loads(line)["task"]) for line in f if line.strip()}

    tasks_parquet = meta_dir / "tasks.parquet"
    if tasks_parquet.exists():
        import pandas as pd

        df = pd.read_parquet(tasks_parquet)
        column = df.index if df.index.name == "task" else df["task"]
        return {normalize(t) for t in column}

    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        instructions = set()
        with episodes_jsonl.open() as f:
            for line in f:
                if not line.strip():
                    continue
                for task in json.loads(line).get("tasks", []):
                    instructions.add(normalize(task))
        return instructions

    raise FileNotFoundError(f"no tasks.jsonl / tasks.parquet / episodes.jsonl in {meta_dir}")

def read_info(meta_dir: Path) -> dict:
    """Return the five fields of meta/info.json, or empty marks when it is absent."""
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        return {}
    info = json.loads(info_path.read_text())
    keys = ["codebase_version", "total_episodes", "total_frames", "total_tasks", "fps"]
    return {k: info.get(k) for k in keys}


def base_stem(stem: str) -> str:
    """Strip the perturbation suffix, so a variant points back at its task."""
    return VARIANT_SUFFIX_RE.sub("", stem)


def is_base_task(stem: str) -> bool:
    """True for a task bddl, False for a perturbed variant of one."""
    return base_stem(stem) == stem


def read_bddl_map(bddl_root: Path) -> tuple[dict, dict]:
    """Map a normalized instruction to the (suite, bddl stem) pairs that carry it.

    Two maps come back: one built from task bddl only, one from every bddl including
    the perturbed variants. LIBERO-plus ships spatial / object / goal as variants
    only, so an instruction from those suites is absent from the first map.
    """
    base_map: dict[str, set[tuple[str, str]]] = {}
    any_map: dict[str, set[tuple[str, str]]] = {}
    for bddl_path in sorted(bddl_root.glob("*/*.bddl")):
        stem = bddl_path.stem
        keys = {normalize(SCENE_RE.sub("", base_stem(stem)).lstrip("_"))}
        match = LANGUAGE_RE.search(bddl_path.read_text())
        if match is not None:
            keys.add(normalize(match.group(1)))
        for key in keys:
            any_map.setdefault(key, set()).add((bddl_path.parent.name, stem))
            if is_base_task(stem):
                base_map.setdefault(key, set()).add((bddl_path.parent.name, stem))
    return base_map, any_map


def scene_names(
    instructions: set[str], base_map: dict, any_map: dict
) -> tuple[set[str], dict[str, str]]:
    """Split the instructions into SCENE-named bddl stems and instructions without one."""
    stems: set[str] = set()
    no_scene: dict[str, str] = {}
    for instruction in instructions:
        hits = base_map.get(instruction) or set()
        scene_hits = {stem for _, stem in hits if SCENE_RE.match(stem)}
        if scene_hits:
            stems.update(scene_hits)
            continue
        variants = any_map.get(instruction) or set()
        if not variants:
            no_scene[instruction] = "bddl に無い"
        elif hits:
            no_scene[instruction] = "SCENE 名でない（" + "/".join(sorted({s for s, _ in hits})) + "）"
        else:
            no_scene[instruction] = "variant の bddl だけ（" + "/".join(sorted({s for s, _ in variants})) + "）"
    return stems, no_scene


def markdown_matrix(names: list[str], sets: dict[str, set]) -> str:
    """Render the pairwise intersection counts, one row per pair."""
    lines = ["| A | B | A のみ | 共通 | B のみ |", "|---|---|---|---|---|"]
    for a, b in combinations(names, 2):
        common = sets[a] & sets[b]
        lines.append(
            f"| {a} ({len(sets[a])}) | {b} ({len(sets[b])}) | "
            f"{len(sets[a] - sets[b])} | **{len(common)}** | {len(sets[b] - sets[a])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="dataset name and its meta/ directory; repeat once per dataset",
    )
    parser.add_argument(
        "--bddl-root",
        type=Path,
        required=True,
        help="bddl_files directory holding libero_10 / libero_90 / ... subdirectories",
    )
    args = parser.parse_args()

    meta_dirs = {}
    for entry in args.meta:
        name, _, path = entry.partition("=")
        meta_dirs[name] = Path(path)

    print("## meta/info.json\n")
    print("| データセット | codebase | エピソード | コマ | タスク | fps |")
    print("|---|---|---|---|---|---|")
    for name, meta_dir in meta_dirs.items():
        info = read_info(meta_dir)
        if not info:
            print(f"| {name} | （info.json なし） | | | | |")
            continue
        print(
            f"| {name} | {info['codebase_version']} | {info['total_episodes']} | "
            f"{info['total_frames']} | {info['total_tasks']} | {info['fps']} |"
        )

    instructions = {name: read_instructions(d) for name, d in meta_dirs.items()}
    names = list(meta_dirs)

    print("\n## 指示文の集合の重なり\n")
    print(markdown_matrix(names, instructions))

    base_map, any_map = read_bddl_map(args.bddl_root)
    scenes = {}
    print("\n## 指示文から bddl へ引けたか\n")
    for name in names:
        stems, no_scene = scene_names(instructions[name], base_map, any_map)
        scenes[name] = stems
        print(
            f"- {name}: 指示文 {len(instructions[name])} 件 → SCENE 名タスク {len(stems)} 個"
            f"／SCENE 名に引けない指示文 {len(no_scene)} 件"
        )
        for instruction, reason in sorted(no_scene.items()):
            print(f"  - {reason}: {instruction}")

    ambiguous = {
        instruction: {stem for _, stem in hits if SCENE_RE.match(stem)}
        for instruction, hits in base_map.items()
        if len({stem for _, stem in hits if SCENE_RE.match(stem)}) > 1
    }
    print(f"\n## 同じ指示文が別の場面にもある（{len(ambiguous)} 件）\n")
    print("**指示文だけでは、どの場面が録られたのかは決まらない。**\n")
    for instruction, stems in sorted(ambiguous.items()):
        print(f"- {instruction} → {', '.join(sorted(stems))}")

    print("\n## SCENE 名タスク（bddl のファイル名）の重なり\n")
    print(markdown_matrix(names, scenes))

    all_scene_stems = {
        stem for hits in base_map.values() for _, stem in hits if SCENE_RE.match(stem)
    }
    covered = set().union(*scenes.values()) if scenes else set()
    print(f"\n- bddl 側の SCENE 名タスク: {len(all_scene_stems)} 個")
    print(f"- 上のデータセットが覆う SCENE 名タスク: {len(covered)} 個")
    print(f"- どのデータセットにも無い SCENE 名タスク: {len(all_scene_stems - covered)} 個")
    for stem in sorted(all_scene_stems - covered):
        print(f"  - {stem}")


if __name__ == "__main__":
    main()
