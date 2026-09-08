#!/usr/bin/env python3
"""指示文の差し替えの組。**先頭が正しい文**で、以降が対照である。

`kind` の意味:

| `kind` | 変えたもの | 期待 |
|---|---|---|
| `true` | 何も（基準） | — |
| `same` | **何も変えない**（基準と同一の文をもう一度） | **0**。ゆらぎの下限 |
| `color` | **色の語だけ** | ← 知りたいのはここ |
| `place` | 位置を表す句だけ | 位置を見ているか |
| `noun` | 物体名 | 反応の上限 |

**T3 の 4 課題は 4 つとも「同じ形で色だけ違う物」の選別を要求する。**
対照として **いま 100% 通っている BASE12 の 2 課題**を置く——
片方は**位置**で見分ける課題（同じ黒椀が複数ある）、
もう片方は**物体名**で見分ける課題（同じ場面に 5 品並ぶ）。
**成功しているのに色に反応していなければ、成功は「色以外で足りている」からである。**
"""

VARIANTS: dict[str, list[tuple[str, str]]] = {
    # ---------------------------------------------- T3（成功率 0%）
    "take_the_red_mug_off_the_left_plate_and_put_it_on_the_table": [
        ("true", "take the red mug off the left plate and put it back on the table behind the other two mugs"),
        ("same", "take the red mug off the left plate and put it back on the table behind the other two mugs"),
        ("color", "take the white mug off the left plate and put it back on the table behind the other two mugs"),
        ("color", "take the yellow and white mug off the left plate and put it back on the table behind the other two mugs"),
        ("place", "take the red mug off the right plate and put it back on the table behind the other two mugs"),
        ("noun", "take the red book off the left plate and put it back on the table behind the other two mugs"),
    ],
    "take_the_yellow_book_off_the_shelf_and_put_it_on_the_table": [
        ("true", "take the yellow book off the top of the shelf and put it back on the table on the left"),
        ("same", "take the yellow book off the top of the shelf and put it back on the table on the left"),
        ("color", "take the black book off the top of the shelf and put it back on the table on the left"),
        ("color", "take the white book off the top of the shelf and put it back on the table on the left"),
        ("noun", "take the yellow mug off the top of the shelf and put it back on the table on the left"),
    ],
    "take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table": [
        ("true", "take the alphabet soup out of the tray and place it on the table in front of the ketchup"),
        ("same", "take the alphabet soup out of the tray and place it on the table in front of the ketchup"),
        ("noun", "take the tomato sauce out of the tray and place it on the table in front of the ketchup"),
        ("noun", "take the cream cheese out of the tray and place it on the table in front of the ketchup"),
    ],
    # ---------------------------------- 対照：BASE12（`rp3000` で 100% 通る）
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": [
        ("true", "Pick the akita black bowl between the plate and the ramekin and place it on the plate"),
        ("same", "Pick the akita black bowl between the plate and the ramekin and place it on the plate"),
        ("color", "Pick the akita white bowl between the plate and the ramekin and place it on the plate"),
        ("color", "Pick the akita red bowl between the plate and the ramekin and place it on the plate"),
        ("place", "Pick the akita black bowl next to the cookies box and place it on the plate"),
        ("noun", "Pick the akita black plate between the plate and the ramekin and place it on the plate"),
    ],
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": [
        ("true", "Pick the alphabet soup and place it in the basket"),
        ("same", "Pick the alphabet soup and place it in the basket"),
        ("noun", "Pick the tomato sauce and place it in the basket"),
        ("noun", "Pick the chocolate pudding and place it in the basket"),
        ("noun", "Pick the milk and place it in the basket"),
    ],
}
