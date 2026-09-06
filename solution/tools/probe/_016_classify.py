#!/usr/bin/env python3
"""016 タスク選択の分類: libero_10 の bddl をパースして、
「単一物体の pick&place かつ開閉なし」を機械的に判定する。

ホスト側で動く（torch 不要、標準ライブラリのみ）。
引き出し先（_016_extract_l10.sh が書いた bddl）を読み、
1 タスクごとに判定の根拠（goal の物体・接触アクション・開閉の有無）を
表で出す。

用法:
    python3 _016_classify.py <extracted_bddl_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

# 対象を「置く」系（逆向きの「戻す」に対応する）と見なす predicate
PLACEMENT = {"On", "In", "At", "InFrontOf", "Behind", "ToTheLeftOf", "ToTheRightOf",
             "OnTopOf"}
# 接触で状態を切り替える系（開閉・スイッチ）。逆向きの「型」が定義しにくいので除外
CONTACT = {"Open", "Close", "TurnOn", "TurnOff", "Press", "Push"}
# 開閉の関与を示す物体タイプ / 名称のキーワード
OC_NAME_HINT = ("door", "drawer", "microwave", "cabinet", "shelf")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            toks.append(c); i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            toks.append(text[i + 1:j]); i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            toks.append(text[i:j]); i = j
    return toks


def parse_sexpr(toks: list[str]):
    pos = 0

    def rd():
        nonlocal pos
        if pos >= len(toks):
            return None
        t = toks[pos]
        if t == "(":
            pos += 1
            out = []
            while pos < len(toks) and toks[pos] != ")":
                out.append(rd())
            pos += 1
            return out
        if t == ")":
            raise ValueError("unbalanced )")
        pos += 1
        return t

    return rd()


def sections(root):
    """root の直下から :regions / :objects / :init / :goal / :fixtures / :obj_of_interest を返す。"""
    out: dict = {}
    for child in root if isinstance(root, list) else []:
        if isinstance(child, list) and child and isinstance(child[0], str):
            key = child[0].lstrip(":")
            if key in ("regions", "objects", "init", "goal", "fixtures",
                       "obj_of_interest", "language"):
                out[key] = child[1:]
    return out


def leaf_preds(nodes):
    """(pred a b ...) 形の葉を全部拾う。pred は大文字始まりのシンボル。"""
    out = []

    def rec(n):
        if isinstance(n, list):
            if n and isinstance(n[0], str) and n[0][0:1].isupper() and len(n) >= 2:
                out.append((n[0], n[1:]))
            for c in n:
                rec(c)

    for n in nodes:
        rec(n)
    return out


def is_oc_involving(pred_args) -> bool:
    """この predicate が開閉に関与しうるか（物体名・predicate 名で判定）。"""
    pred = pred_args[0]
    if pred in CONTACT:
        return True
    for a in pred_args[1:]:
        if isinstance(a, str) and any(h in a.lower() for h in OC_NAME_HINT):
            return True
    return False


def analyze(path: Path):
    root = parse_sexpr(tokenize(path.read_text(encoding="utf-8")))
    sec = sections(root)
    init_preds = leaf_preds(sec.get("init", []))
    goal_preds = leaf_preds(sec.get("goal", []))

    oc = [p for p in goal_preds if is_oc_involving(p)]
    contact_in_init = [p for p in init_preds if is_oc_involving(p)]
    objects = [a[1] if len(a) >= 2 and isinstance(a[1], str) else a
               for a in sec.get("objects", [])]
    # goal が単一物体の置く系か（On/In など 1 物体 1 領域）
    placement_goals = [p for p in goal_preds if p[0] in PLACEMENT]
    single_object = (len(placement_goals) == 1
                     and all(not is_oc_involving(p) for p in goal_preds))

    return {
        "stem": path.stem,
        "language": (sec.get("language", ["?"])[0]
                     if sec.get("language") else "?"),
        "objects": objects,
        "n_objects": len(objects),
        "goal_preds": goal_preds,
        "contact_in_goal": oc,
        "contact_in_init": contact_in_init,
        "has_oc": bool(oc or contact_in_init),
        "single_object": single_object and len(objects) >= 1,
        "obj_of_interest": sec.get("obj_of_interest", []),
        "n_regions": len(sec.get("regions", [])),
    }


def main():
    d = Path(sys.argv[1])
    rows = []
    for p in sorted(d.glob("*.bddl")):
        try:
            rows.append(analyze(p))
        except Exception as e:  # noqa
            rows.append({"stem": p.stem, "language": f"PARSE_ERROR {e}",
                         "objects": [], "n_objects": 0, "goal_preds": [],
                         "contact_in_goal": [], "contact_in_init": [],
                         "has_oc": True, "single_object": False,
                         "obj_of_interest": [], "n_regions": 0})

    print(f"\n=== {len(rows)} タスク（libero_10 元タスク・SCENE 名）===")
    print(f"{'適':4} {'物体数':>5}  タスク名 / 指示文")
    print("-" * 100)
    for r in rows:
        ok = "○" if (r["single_object"] and not r["has_oc"]) else ("△" if not r["has_oc"] else "✗")
        note = []
        if r["has_oc"]:
            oc_names = sorted({a[1] for a in r["contact_in_goal"] + r["contact_in_init"]
                               if len(a) >= 2 and isinstance(a[1], str)})
            note.append(f"開閉: {','.join(oc_names) or r['contact_in_goal'][:1]}")
        if not r["single_object"]:
            gl = r["goal_preds"]
            note.append(f"goal={len(gl)}件(単一でない)")
        print(f"{ok:4} {r.get('n_objects','?'):>5}  {r['stem']}")
        print(f"{'':4} {'':>5}    指示: {r['language']}")
        if note:
            print(f"{'':4} {'':>5}    { ' | '.join(note) }")
        print(f"{'':4} {'':>5}    obj_of_interest: {r['obj_of_interest']}")
        print()

    chosen = [r["stem"] for r in rows if r["single_object"] and not r["has_oc"]]
    print("\n=== 候補（単一物体 pick&place かつ開閉なし）===")
    for s in chosen:
        print(" ", s)
    print(f"\n合計 {len(chosen)} 件")


if __name__ == "__main__":
    main()