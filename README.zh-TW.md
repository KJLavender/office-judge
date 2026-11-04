# office-judge

[English](README.md) | **繁體中文**

office-judge 讀員工每天寫的信件、聊天訊息和會議記錄，做三件事：擋下洩漏機密的訊息、把客服工單分給對的團隊、抓出待辦事項。它處理的是繁體中文。

每個決定都是一個是非題或選擇題，交給「評審」判斷。評審是一條串起來的鏈：先用便宜的規則，再用小型 decision model（Jev 或開源的 Kev-4B），小模型沒把握的才交給通用的 LLM（Claude），最後是人工審核佇列。這條鏈寫在 `judge-core`，是放在這個 repo 裡的一個小函式庫。

![Demo：間接外洩被抓到；不確定的句子依序經過規則、Kev-4B、Claude、人工佇列](docs/demo-indirect.png)

## 成果

以下是繁體中文的外洩防護（DLP，data loss prevention）結果。公司資料全是虛構的，訊息和標註都是 LLM 產生的。

| | 結果 |
|---|---|
| 整條鏈（規則、Kev-4B、Claude、人工），100 則獨立驗證訊息 | 56 則外洩全部攔下，準確率 0.93，42% 的句子需要問 LLM |
| Kev-4B 門檻在一份資料上調好、凍結，套到新資料 | 準確率 0.88（在新資料上重調是 0.89），所以門檻可以沿用 |
| Kev-4B 在 RTX 3060 Ti 上的延遲 | 每句 p50 62 ms，使用者按下送出前就能檢查完 |
| 小型 encoder decision model（Julia-1、Laya 英文版與多語版） | 繁中不可用：AUC 0.66–0.80，而且題目換個寫法結果就變 |

AUC 是模型把外洩排在非外洩前面的能力（1.0 是完美，0.5 等於亂猜）。

![各模型 AUC](docs/model-auc.png)

這兩點是我學到、也改變了設計的事：

- 門檻只對一個模型、一種題目寫法有效。Kev-4B 的排序幾乎完美（AUC 0.98），但用預設的 0.5 當門檻會漏放 42% 的外洩。題目改幾個字，機率就偏移到每一句都要送去問 LLM。所以每個 policy 檔有各模型自己的 `model_thresholds`，題目文字也固定成調門檻時用的寫法。`scripts/bench_dlp.py` 會比較不同模型和寫法，輸出在 `data/out/bench_dlp.json`。
- 已知格式先遮罩，模型永遠看不到原文。規則會抓台灣身分證、居留證、統一編號（含檢查碼）、信用卡號、手機號碼、API 金鑰和專案代號，換成之後可以還原的 placeholder。評審模型收不到真實內容。

標註還需要人工檢查。`data/out/human_labels.csv` 放了 40 句要給人標的句子，目前一句都還沒標。

## 三件事

| 工作 | 做什麼 | 做法 |
|---|---|---|
| 外洩防護（DLP） | 把關訊息 | 先用規則遮罩已知格式，再用一次 Jev 請求逐句判斷。結果是放行、遮罩後放行（可還原），或擋下並說明原因 |
| 工單分流 | 選團隊、判斷急迫度 | 一次 Jev 請求同時問「哪個團隊」（選擇題，各團隊的負責範圍寫在選項說明裡）和「多急」（打分數） |
| 待辦萃取 | 找出待辦和負責人 | 一次 Jev 請求對每一句問「是不是待辦」和「負責人是誰」（與會者，或「未指定」），附前後句當上下文。被挑出來的句子才交給 LLM 換算期限 |

Jev 是 TypeSafe 的 decision model。它一次請求可以回答很多題，每題回一個機率。Kev-4B 等開源模型可以透過 llama.cpp 提供同樣格式的請求。

每個問題走的路：

```
規則 --命中--> 決定
  | 沒命中
Jev（所有問題一次送）--信心夠--> 決定
  | 灰色地帶
LLM（灰色的問題一次送）--信心夠--> 決定
  |
人工佇列（把關類的問題在審核前維持擋下）
```

每一層都會寫進 SQLite 稽核紀錄：輸入雜湊、各層的答案與機率、模型版本、延遲、成本。

## 執行方式

需要 Python 3.11 以上和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra eval
uv run pytest                                  # 53 個測試
uv run uvicorn office_judge.app:app --reload   # http://127.0.0.1:8000
```

沒有 API key、也沒有本地模型時，網頁會用關鍵字規則模擬判斷，所以不用任何模型也能把整個流程跑一遍。後端由環境變數決定：

| 變數 | 說明 |
|---|---|
| `TYPESAFE_API_KEY` | 用 TypeSafe 的 Jev 當主要評審 |
| `TYPESAFE_BASE_URL` | 指向另一個協定相同的伺服器，例如 [jev-rs](https://github.com/yijunyu/jev-rs) 或 llama.cpp，就能完全在本機跑 |
| `TYPESAFE_DEFAULT_MODEL` | 預設 `jev-1.14.0`。alias 會移動，請綁定版本 |
| `OFFICE_JUDGE_PRIMARY` | `jev`、`llm` 或 `demo`。預設：有設 key 或 base URL 就是 `jev`，否則是 `demo` |
| `OFFICE_JUDGE_FALLBACK`、`OFFICE_JUDGE_WRITER` | `on` 或 `off`。分別是 LLM 第二層，以及由 LLM 寫擋下理由、換算期限。有安裝 `claude` 指令時兩者預設都開 |
| `JUDGE_LLM_MODEL` | 傳給本機 Claude Code CLI 的模型，預設 `sonnet` |

LLM 這層呼叫本機的 `claude -p` 指令，用的是你的 Claude 登入，不需要 API key。

### 用本地模型跑 demo

`./scripts/demo.sh` 會先透過 llama.cpp 啟動 Kev-4B，再啟動網頁並開啟頁面。顯示卡有 4 GB 以上空閒 VRAM 就用 GPU，否則用 CPU（每句約 1 秒）。`./scripts/demo.sh --offline` 完全不用模型，改用關鍵字規則模擬。這個腳本用 `cmd.exe` 開瀏覽器，所以預期在 Windows 或 WSL 上執行。

實際啟動模型的是 `scripts/serve_local.sh`。它不會幫你下載任何東西，需要事先準備好：

- llama.cpp 的 server 執行檔：`.cache/llama/cpu/oj-llama-server.exe`（GPU 用 `.cache/llama/cuda/llama-server.exe`）
- 模型檔放在 `~/Desktop/IEKA/.cache/llama/models`（可用 `IEKA_MODELS` 改路徑），例如 `Kev-4B-Q4_K_M.gguf`

只啟動 Kev-4B，並讓網頁接上它：

```bash
KEV_GPU=1 ONLY=kev-4b ./scripts/serve_local.sh &
TYPESAFE_BASE_URL=http://127.0.0.1:8195 TYPESAFE_DEFAULT_MODEL=kev-4b uv run uvicorn office_judge.app:app
```

在這份資料上只有 Kev-4B 堪用，它的門檻寫在 `src/office_judge/policies/dlp.yaml` 的 `model_thresholds`。

## 評估

```bash
uv run python scripts/toy_eval.py                                # 20 筆人工標註資料：門檻掃描＋校準曲線
uv run python scripts/toy_eval.py --backends jev llm --cascade   # 比較後端、算 LLM 救援率
uv run python scripts/jev_checks.py                              # Jev 實測：繁中、合併送出、選項順序
uv run python scripts/gen_data.py dlp --n 300                    # 產生虛構公司的合成標註資料
uv run python scripts/validate_pipeline.py --data data/out/dlp_val.jsonl   # 讓獨立驗證資料走完整條鏈
```

LLM 救援率是指：灰色地帶裡，LLM 推翻 Jev 而且推翻正確的比例。依賴 LLM 這一層之前先量一下：如果兩個模型常錯在同樣的句子上，第二層就幫不上什麼忙。

## 目錄

```
src/judge_core/      Judge 介面、Jev／LLM／Rule／Human／Fake 後端、串接、policy、稽核、評估
src/office_judge/    偵測器、文字處理（斷句、可還原遮罩）、三件工作、policies/*.yaml、company.yaml、網頁
scripts/             demo.sh、serve_local.sh、toy_eval、jev_checks、gen_data、bench_dlp、validate_dlp、validate_pipeline
data/toy_dlp.jsonl   20 筆人工標註句子
data/out/            100 則的調參與驗證資料、benchmark 輸出、人工標註用的表格
```

所有公司、人名與個資皆為虛構。
