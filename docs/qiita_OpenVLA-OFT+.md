---
title: PARC2026でOpenVLA-OFT+を試した記録
tags: PARC2026 OpenVLA Robotics 機械学習 QiitaCLI
author: arugo11
slide: false
---

[PARC2026でOpenVLA-OFT+を試した記録](https://qiita.com/arugo11/private/bba6dd7d2bf90963c4de)

## 前書き

やらかした...[コミケでLLM本頒布したり](https://x.com/argo_dev_/status/2086347994400764035),VLAの研究してたらVLAのコンペのレポート出し忘れた......
せっかくA100で長時間データ生成,学習,評価を行ったので結果を供養

## 今回やったこと

PARC2026に向けて,LIBERO-Plusで学習済みのOpenVLA-OFT+をベースに,5 task,3,900 episodeの追加データでLoRA学習を行いました.
LLMとvision backboneにはLoRAを適用し,action headとproprio projectorも学習しています.

追加データには,背景と照明を変えた成功例に加えて,把持失敗や落下後にexpertが成功まで戻す軌跡を含めました.
推論では外部カメラと手首カメラ,proprio,instructionを入力し,7次元actionを8 stepまとめて生成します.

学習後の方策は,公式評価とは別に構築したLIBERO環境で80 episodeを実行し,suiteとtask family別に成功率を確認しました.
生成した追加データは,条件別に分析できるメタデータとRLDSをまとめてHugging Faceで公開しています.

## Adapter-checkの結果

公式評価とは別に構築したLIBERO環境で80 episodeのadapter-checkを行ったところ,70 episode,87.5%が成功しました.
suite別では`libero_goal`と`libero_object`が20/20,`libero_spatial`が19/20,`libero_10`が11/20でした.
さらに`libero_10`をtask family別に分けると,`multi_object_sequence`は6/12でした.
この結果から,長い操作系列を追加調査の対象にしました.

![W&B OpenVLA-OFT adapter-checkのtask family別success rate](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1168086/d02e72dd-f6d1-4f94-ba1c-264d8b589443.png)

*図1. W&Bの80 episode adapter-check.*

ただし,この比較ではsuiteとtaskが揃っていません.
したがって,6/12という値だけで操作系列長を失敗原因と断定することはできません.
`stove_toggle`も1/2ですが,2 episodeでは傾向を判断できないため除外しました.

次に試すなら,単一操作,接触を伴う操作,多段階操作でepisode数を揃えます.
この分け方なら,系列長と接触の難しさを別々に比較できます.

## OpenVLA-OFT+を選んだ理由

ベースには[`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata)を選びました.
この重みにはLIBERO-Plusによるmix-SFTが施されており,連続action head,action chunk,proprio入力をそのまま使えます.

[LIBERO-Plusの公開結果](https://huggingface.co/datasets/Sylvest/LIBERO-plus)では,OpenVLA-OFT+は総合79.6で,camera 92.8,light 94.9,noise 89.3でした.
一方,robot initial stateは30.3でした.
カメラ,照明,ノイズには強いものの,初期姿勢の変化には弱いという偏りがあります.
この偏りが公開結果から分かっていたため,追加データを初期姿勢,空間配置,失敗後の補正へ集中させられました.

素のOpenVLAから同じ推論方式を作る場合は,連続action head,action chunk,proprio入力を追加学習する必要があります.
そこで,モデルの推論機構を作り直すよりも,データ生成と失敗条件の追加学習へ計算時間を使う方がよいと判断しました.

## 学習と推論

### モデルと推論

| 項目 | 設定 |
| --- | --- |
| ベースモデル | [`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata) |
| revision | [`a85655ec941bae6644c9fbdf62db02b9726d7cf5`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/a85655ec941bae6644c9fbdf62db02b9726d7cf5) |
| 入力 | agentview,wrist camera,8次元proprio,instruction |
| 出力 | 7次元actionを8段,shapeは`(8, 7)` |

![推論policyの構成](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0%2F1168086%2F16d535fe-b442-4fba-9945-33b2541dddce.png)

*図2. 2画像,8次元proprio,instructionから8段の7次元actionを出力する構成.*

推論時は2画像を180度回転し,center cropを適用します.
状態入力はEEF位置,axis-angle,gripper qposから作る8次元stateです.
action headは8 actionをまとめて生成し,policy serverはqueueから1 actionずつ`float32 (7,)`で返します.

action chunkを長くすると推論回数は減りますが,後半のactionほど古い観測に基づくことになります.
長いタスクの後半で軌道がずれる場合は,horizon 8と2または4を比較すると,再推論頻度の影響を切り分けられます.

今回の推論実装ではinstruction中の語からsuiteを選び,suite別のaction normalizationを適用しました.
action tableや有限状態機械は使っていません.
ただし,suite選択は手書きのtask依存処理なので,別の課題へ移植する場合は外すか,改めて検証する必要があります.

adapter-checkの推論時間はA100上で最大約518 msでした.
[公式実装](https://github.com/matsuolab/PARC2026_pre#推論時間制限)では,chunkを生成する重いリクエストにも10秒の上限があります.
ただし,このrunではL4上の推論時間を測っていません.

### 追加学習の設定

| 項目 | 設定 |
| --- | --- |
| 学習方法 | LoRA + action head + proprio projector |
| 学習対象 | LLMとvision backboneのLoRA,action head,proprio projector |
| LoRA | rank 32,alpha 16,dropout 0.0 |
| batch | 1,gradient accumulation 8 |
| learning rate | `5e-5`,warmup 200,補助ヘッド倍率2.0 |
| optimizer | AdamW + MultiStepLR |
| precision | bfloat16 |

学習データは5 task,3,900 episode,688,747 stepsです.
train 3,510,validation 195,test 195へ分割し,データの役割を4区分に分けました.

| suite | task数 |
| --- | ---: |
| `libero_goal` | 2 |
| `libero_object` | 1 |
| `libero_spatial` | 2 |

| データ区分 | episode | 役割 |
| --- | ---: | --- |
| broad-success | 1,650 | suiteをまたぐ成功例 |
| domain-randomized | 1,350 | BackgroundとLightの条件変化 |
| recovery | 600 | 失敗後の補正と最終成功 |
| canonical | 300 | 追加データとの比較基準 |

![3,900 episodeのデータ構成](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1168086/7bdc24b1-204e-45e8-a98b-8f86b566e2b0.png)

*図3. 学習データの4区分.*

canaryはstep 2,000でvalidation loss 0.0763,current action L1 0.0804,next actions L1 0.0757に到達しました.
しかし,validation指標はstep 800,1,400,1,800付近で一時的に悪化しています.
training lossだけを見てcheckpointを選ぶと,この揺れを見落とします.

![W&B C1 canaryのvalidation curve](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1168086/dab5a967-986b-4e65-81d9-ff6e1921d053.png)

*図4. C1 canaryのvalidation指標.*

## 失敗と切り分け

| 症状 | 原因として調べた箇所 | 対応 |
| --- | --- | --- |
| 把持点がずれる | camera orientation,crop,resize | 画像変換とcamera順序を固定する |
| 初手は正しいが後半でずれる | action chunkと再推論頻度 | horizon 2,4,8を比較する |
| 把持と解放だけ失敗する | gripper normalization,sign | gripperを単独指標で追う |
| 見た目は完了しているがtimeoutになる | stage completion predicate | task successとcollisionを分けて記録する |
| 長い系列の後段で崩れる | 多段階データの不足 | failure prefixから補正actionと成功までを学習する |

80 episodeのadapter-checkでは,Goalのpushが2/2,drawer-openが4/4でした.
ところが,条件を変えてデータを生成すると,pushの接触方向とdrawer-openのcollisionが安定しませんでした.
この2 taskは追加データの対象から外しています.

成功率が低いtaskをすぐ増量すると,シミュレータ側の不安定さまで学習データへ入ります.
先に成功判定と接触条件を個別に確認した方が,無駄な生成と学習を減らせます.

## おまけの公開データセット

失敗条件の診断と追加学習に使った[OpenVLA-OFT+ Robustness and Recovery](https://huggingface.co/datasets/argo11/openvla-oft-plus-robustness-recovery)も公開しています.
弱点の観測とデータ生成には公式評価とは別のLIBERO環境を使い,公式評価環境の観測や評価結果は学習へ入れていません.

公開物は2種類あります.
Dataset Viewerではepisode単位の条件を確認でき,RLDSではstepごとの画像,proprio,actionを読めます.

| 公開物 | 件数 | 用途 |
| --- | ---: | --- |
| Dataset Viewer | 3,900 episode,42列 | condition,family,recovery stageの集計 |
| RLDS main | 3,380 episode | 通常の追加学習 |
| RLDS extension | 520 episode | 条件変化と失敗後の補正を分けた比較 |

![episode単位のメタデータとstep単位の軌跡の対応](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0%2F1168086%2F65c7ab8c-dda5-4e24-99e5-076b007e3c50.png)

*図5. 全3,900 episodeに占めるmainとextensionの比率.右側は,[公開sample metadata](https://huggingface.co/datasets/argo11/openvla-oft-plus-robustness-recovery/raw/main/assets/sample-recovery-episode.json)の1行とstep 0から188までのtrajectoryとの対応.*

### 観測した弱点

各suiteを60 episodeずつ評価したところ,Spatialのsafe successは63.3%,collisionは35.0%でした.
Goalは90.0%と8.3%,Objectは95.0%と5.0%です.
この差を受けて,Spatialの配置と接触を優先して追加データを作りました.

この順位は,OpenVLA-OFT全体へ一般化できるものではありません.
同じepisode数で3 suiteを比べたときに観測した,今回の条件差です.

![公開データセットが切り出した弱点](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1168086/2eba134d-e9e6-45cd-8fc6-e0bde20b69fe.png)

*図6. suite別safe successとcollision.公開データセット内の`weakness-observations.csv`に基づく.*

3,900 episodeは5 taskの条件違いであり,一般的なロボット操作を広く覆うデータではありません.
condition別ではClean 2,550,Background 676,Light 674です.
task family別では`pick_place` 3,288,`stove_toggle` 384,`multi_object_sequence` 228で,pick-placeへ大きく偏っています.

この偏りがあるため,全体平均だけではBackground,Light,multi-objectの悪化が埋もれます.
validationにもconditionとtask familyの区分を残す必要があります.

![条件とtask familyの実データ構成](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/1168086/8b1842bb-d0ee-4de8-be32-9fb76f93050d.png)

*図7. condition別のtask family構成.公開データセット内の`condition-by-family.csv`に基づく.*

### 失敗後の補正データ

失敗後の補正を含む600 episodeの内訳は,drop 300,grasp failure 220,lift failure 40,approach miss 20,placement failure 20です.
各episodeには失敗時点の画像だけでなく,失敗までの軌跡,expertによる補正,最終成功までの軌跡を連続して保存しました.

#### expert補正の中身

ここでいうexpert補正は,失敗episodeへ`recovery_stage`ラベルを付ける処理ではありません.
まず,方策が失敗状態へ入るまでの軌跡を残します.
次に,その失敗状態からexpertがタスクへ戻すaction列を接続し,success判定までを同じepisodeとして保存します.
補正actionは失敗の種類に応じて,再接近,再把持,持ち上げ直し,再配置へ変わります.

`recovery_stage`はDataset Viewerでepisodeを抽出し,比較するためのメタデータです.
モデルへ与える教師信号は各stepのactionです.
今回の[OpenVLA-OFT](https://github.com/moojink/openvla-oft)は,時刻`t`の観測から8 stepのaction chunkを予測します.
そのため,失敗後の観測`o_t`と,復帰方向を示す`a_t ... a_{t+7}`を対応づけて学習できます.

成功例だけでbehavior cloningを行うと,把持位置を外した後の観測が学習データにほとんど現れません.
当然,その状態から戻るactionにも教師がありません.
expert補正を加えると,失敗を検出する分類器を作らなくても,ずれた状態から成功軌道へ戻るactionを直接学習できます.

![expert補正によって増える教師信号](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0%2F1168086%2F9068e763-e988-4dcc-a754-b25a637bea05.png)

*図8. 失敗で軌跡を打ち切る場合と,失敗後の観測へ8-step action chunkを対応させて最終成功まで接続する場合の違い.画像には[公開contact sheet](https://huggingface.co/datasets/argo11/openvla-oft-plus-robustness-recovery/blob/main/assets/figures/sample-recovery-episode.png)の実フレームを使用.*

補正データにも限界があります.
特定の失敗姿勢に対する補正を増やしても,未知の失敗全般から復帰できるとは限りません.
追加学習後は全体successに加えて,drop,grasp failure,lift failureなどの種類別に復帰率とcollisionを測る必要があります.

公開sampleは`libero_spatial`の`grasp_failure`です.
step 0の開始,step 63の把持失敗後,step 126の補正中,step 188の最終成功を記録しています.
2つのカメラを並べると,外部カメラでは腕全体の移動を,手首カメラでは対象物との位置関係を追えます.

![公開sample recovery episodeの外部カメラと手首カメラの実フレーム](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0%2F1168086%2Fcd770b2e-9005-4e43-95e7-23f54f15d443.png)

*図9. [公開contact sheet](https://huggingface.co/datasets/argo11/openvla-oft-plus-robustness-recovery/blob/main/assets/figures/sample-recovery-episode.png)の実フレームを2視点,4時点で再配置.下段のpixel差分は,step 63から126の姿勢調整と,step 126から188の配置変化を強調している.*

### データセットを使う順序

1. 最初にSpatial,Goal,ObjectとClean,Background,Lightを分けて評価する.
2. 悪化したcondition,family,recovery stageだけをDataset Viewerで選ぶ.
3. RLDSの`main`と`extension`を分けて追加学習し,混ぜないrunと比較する.
4. 全体lossではなく,condition,family,recovery stageごとにcheckpointを選ぶ.

RLDSは次のコマンドで取得できます.

```bash
hf download argo11/openvla-oft-plus-robustness-recovery --repo-type dataset --revision v1.0.0 \
  --include "rlds/**" --local-dir openvla-oft-robustness-recovery
```

公開データは最終的に成功へ到達したtraining trajectoryです.
このデータ自体は,追加学習によって失敗率が改善したことを示すbenchmark結果ではありません.
効果を確かめるには,追加学習の前後でsuite,scene,conditionを揃え,方策を再評価する必要があります.

## 権利関係

| 対象 | ライセンスまたは記載 | 出典 |
| --- | --- | --- |
| OpenVLA-OFT+ model repository | MIT | [model card](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata) |
| LIBERO-Plus Hub dataset | Dataset Card metadataはMIT | [dataset card](https://huggingface.co/datasets/Sylvest/LIBERO-plus) |
| LIBERO-Plus sourceと固有assets | 独立したLICENSEを確認できず,原著作者への確認が必要 | [repository](https://github.com/sylvestf/LIBERO-plus) |
| 公開した追加データ | `other`,データセット全体の利用条件は未確定 | [dataset card](https://huggingface.co/datasets/argo11/openvla-oft-plus-robustness-recovery) |
| OpenVLA-OFT code | MIT | [repository](https://github.com/moojink/openvla-oft) |
| Llama 2 7B | Llama 2 Community License | [model card](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| Transformers code | Apache License 2.0 | [LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| dlimp code | 明示ライセンスがなく,再配布許諾を確認できない | [repository](https://github.com/kvablack/dlimp) |

# 結論

8/18締め切りということは, 8/18までおkということではない

# あとがき

SigLIP周りはpre-processingやfeature extractionが少しずれるだけでも影響しますし, 成功軌道だけの学習では一度失敗した後に戻れないので. この2点はかなりスコアに響いた可能性がありそう

失敗直後の観測と補正actionをセットで入れると効果が大きいのではと思います.
