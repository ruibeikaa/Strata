# 部分 B：M1–M4 联调（深度 2：N2 / B0）—— 2026-10-19

工作树 `builds\strata-p1b`（分支 `p1-b`，HEAD `b227571e` = daeed844 + M1 `11ec28f7` + M2 `caf0612b` + M3a `b227571e`）。

> **修复轮（第 7 节，同日晚）**：两份复核（`B-review-identity.md`、`B-review-engineering.md`）的 5 条 major 已全部改完，
> p1-b 上新增 3 个代码提交（M1 `7efe310d`、M2 `cda5afcb`、M3 `5671ad9f`），M4 工具同步改了。**第 4 节的 G2 判据按第 7.4
> 节读**：状态摘要只在同一个 `end` 时比较，每臂多 3 个请求（含停放后续写）。被测 exe 换成修复轮的产物（sha256 前 16 位
> `83bcd8d38ca73e69`）。下面第 0–6 节保留联调当时的原文。
- 引擎代码这一轮**一行没改**：联调里查出来的两处不一致都在 M4 的 Python 工具里，已修。（修复轮改了引擎，见第 7 节。）
- GPU 只用了板 B 的 die 4（19:00）和 die 5（1A:00），用之前 `nvidia-smi --query-compute-apps` 查过，两颗都空着，没有等。
- 路由一个没碰。生产文件（`engine-sm60`、`scripts\`、`runtime-state`）没碰。

## 0. 结论

| 项 | 结果 |
|---|---|
| 构建 `strata`、`pl_sched_test`、`mtp_chain_parity` | `BUILD_OK`，0 个错误。`strata.exe` 已是 HEAD 的产物，ninja 没有要重编的；sha256 前 16 位 `93857b6ae1972fc7` |
| `pl_sched_test` 全量复跑 | **PASS，0 失败**，326 s，退出码 0。(d) N2 对旧循环：802,830 个世界逐行相同，0 个不同 |
| `mtp_chain_parity` 复跑（板 B） | die 4 种子 1 × 300 轮 **PASS**；die 5 种子 7 × 400 轮 **PASS**；七项检查都是 0 个不一致 |
| (a) 旧 pw-2 循环 | daeed844 第 10482–11113 行（632 行，41,376 字节）在新文件里**逐字节出现 1 次**；没有任何 diff 操作落在这段里面 |
| (b) 不设开关时 | 参数、显存账、启动日志、每请求路径、timing 三行都和今天相同，见第 2 节逐条核对 |
| (c) 真机冒烟 | 单颗 die 装不下模型，按要求不做；8190 的命令写在第 4 节，由调用方跑 |
| 联调查出的问题 | 2 处，都在 M4 工具里，已修，见第 3 节：(1) 第四行统计的正则对不上 M3a 实际打印的格式，`restage` 判据会一直不打勾；(2) trace 里 `# mode=B0` 这一行被当成新请求的开头 |

- **能不能上 8190 跑 G2**：可以。第 4 节的两条命令就是 G2。
- **G3 现在不能跑**：M3a 的 `--pipeline-windows 3` 仍按 B0 跑，X 臂量到的是 B0。要等 M3b。

## 1. 复跑的测试

### 1.1 构建

- 命令：`builds\build-strata-p1b.bat strata pl_sched_test mtp_chain_parity`。日志在 `B-INT-build.log`。
- 这次只重编了 `mtp_chain_parity` 的旧版对照源（`mtp_base.cpp`）。原因是配置阶段每次都会用 `git show daeed844` 重新生成它，属于正常现象。
- `strata.exe`（18:10）和 `pl_sched_test.exe`（17:44）都已经是当前源码的产物。

### 1.2 `pl_sched_test`（M2 的符号测试，纯 CPU）

- 命令：`strata-p1b-build\pl_sched_test.exe`（全量，不加参数）。
- 输出：`B-INT-work\pl_sched_test_full_out.txt`。
- 结果：
  - (a) 10-17 trace 输入：N2 和 trace 里的裁决有 1250/1280 个相同（a, on），和 M2 报的一样。另外 30 个是符号世界每个位置只留一个猜测造成的，不是错误。
  - (b) 随机输入：每种模式约 100 万次裁决，0 失败。
  - (c) 穷举：每种模式 1,119,744 次运行，0 失败。
  - (d) N2 对旧循环的逐句转写：802,830 个世界逐行相同。
  - 合计：`PASS (0 failures)`，退出码 0。

### 1.3 `mtp_chain_parity`（M1 的草稿链对拍，板 B）

- 用 M1 的同一个运行脚本：模型是 swift-abl-iq3_s，草稿头 rt-zs-v2，`kv int8`，窗口 8192，9000 格提示。

| die | 种子 × 轮数 | 1. n_out ≤ 7 新=旧 | 2. 11 输出前缀 | 3. 分步=一次排满 | 4. source_done | 5/6 | 7. 串行 draft() | 结果 |
|---|---|---:|---:|---:|---:|---|---:|---|
| 4（19:00） | 1 × 300 | 1199 | 1199 | 2780 | 300 | ok | 56 | **PASS** |
| 5（1A:00） | 7 × 400 | 1597 | 1597 | 3615 | 400 | ok | 56 | **PASS** |

- 日志：`B-INT-work\parity_seed1_d4.log`、`parity_seed7_d5.log`。
- 链的主机墙钟：每条 11 输出链 19.3–19.5 ms，和 M1 一样。

## 2. 一致性核对

### 2.1 (a) 旧 pw-2 循环逐字节不变

- 脚本：`B-INT-work\check_oldloop.py`，输出 `check_oldloop_out.txt`。
- 做法：
  1. 从 `git show daeed844:src/program/generate.cpp` 里找到旧循环的开头 `if (pl_want && pl_serial == nullptr && !cancelled && produced_n < max_new) {`；
  2. 按大括号配对找到结尾（去掉字符串和注释再数）；
  3. 在新文件里数这整块出现的次数；
  4. 再对两份文件做逐行 diff，看有没有操作落在这一段里。
- 结果：
  - 第 10482–11113 行，共 632 行、41,376 字节，在新文件里出现 **1 次**。
  - 落在这一段里的 diff 操作：**无**。最近的两处：
    - 段前：新循环整块插在原第 10481 行之后（新文件第 10836–11268 行）；
    - 段后：`CHECK_STATE` 转储插在原第 11283 行之前。
- 段外改过一行条件：
  - `pl_want` 的定义（原第 10472 行）改成 `pipe && pl_pw >= 2 && pl_snap_ok && !pl_new`。
  - `pl_serial` 改成 `!pl_want && !pl_new ? nullptr : …`。
  - 开关不设时 `pl_mode = -1`，所以 `pl_new = false`，两个表达式和原来逐值相同。

### 2.2 (b) 不设开关时什么都不变

逐个 grep 了新开关的读法和守卫（`generate.cpp` 当前行号）：

| 位置 | 新东西 | 不设开关时 |
|---|---|---|
| :1897 | `--pipeline-windows` 钳位改成 ≤ 3 | 启动脚本传的是 2，结果仍是 2。`scripts\` 里没有任何地方传 3，也没有出现任何新开关名（已 grep） |
| :3108–3109 | `STRATA_PIPELINE_SLOTS`、`STRATA_PIPELINE_UNDO_PER_STAGE` 一直读（不要求 DEBUG） | 都没设 → `pl_ns = 2` |
| :3326–3327 | `kPipeWindowMib = 160 × (pl_ns − 1)`，`kPipeSnapMib = 48 × pl_ns` | pl_ns 2 → 160、96，和今天相同 |
| :4382–4391 | `pipe_first`；启动日志 | pl_ns 2 → `pl_ns × 快照` = 今天的 2 份。打印的仍是原来那行 "216 MiB … (the second verifier, the GDN snapshots)"，新格式只在 pl_ns > 2 时打 |
| :6943 起 | 额外槽的验证器和 hand-off | `if (pipe && pl_ns > 2)` 挡住，不分配 |
| :7052 | 额外槽的 plan sink | `q < pl_ns` 的循环，pl_ns 2 时一次也不跑 |
| :7121、:7154 | 额外槽的快照；"slot pool holds" 一行 | `pl_ns > 2` 挡住 |
| :7173、:7198 | `pl_prepare`、`pl_drain` 的 `par < pl_ns` | pl_ns 2 = 原来的 `par < 2`；`PSDq[0..1]` = 原来的 `PSD[0..1]` |
| :7222 | `CHECK_STATE` | `pipe_dbg_env`，要 `STRATA_PIPELINE_DEBUG=1` |
| :10186–10195 | `NEWLOOP`、`THETA3`、`FORCE_MISS_C` | `pipe_dbg_env`，要 DEBUG |
| :10209–10220 | 开关文件的新键和回显 | 整块在 `STRATA_PIPELINE_SWITCH` 里面（要 DEBUG）。回显只在原来那行末尾追加 4 项 |
| :10225 起 | 每请求的 `pl_mode` | 条件 `pl_pw >= 3 \|\| pl_ups \|\| pl_newloop` 全假 → −1 → 旧循环 |
| :10622、:10631、:10635 | 额外槽的 `set_sampling`、`pcie_num`、`set_history` | `q < pl_ns` 的循环，不跑 |
| :11157、:11216–11218 | `mtp.set_source_done` / `chain_close` | 只在新循环分支里。`set_chain_max` 在 `generate.cpp` 里没有调用 |
| :12074 | `CHECK_STATE` 转储 | `pl_check_state` 挡住（DEBUG）；在 `decode_ms` 取完之后才做，不进 serve 行的 tok/s |
| 结构体 | `PlDiag::v[2][5]`、`PlSnap` 数组 `[5]`、`GpuStage::ver_x[3]`（空指针） | 旧循环只写 [0]、[1]。看门狗打印跳过空指针，槽 0、1 的名字不变 |

- **草稿器（M1）对默认路径的两处影响**，都已证明不改结果，或只在边界上改草稿：
  - 暂存行的布局和设备 arena 的偏移后移了。对拍第 1、7 项（新旧代码 n_out ≤ 7、串行 `draft()`）逐位相同。
  - `n_out` 按 `max_cells` 钳位：只在上下文最末尾触发。那里旧代码会把草稿 KV 写越界，所以只影响草稿，不影响回复。
- **一处要知道的**：`CHECK_STATE` 读状态失败时会打 `ERR` 并 `return 1`，也就是结束引擎。这只在 DEBUG 开关下出现，生产路径不会走到；8190 的 A/B 里如果引擎中途退出，先看这一条。

## 3. 联调查出并修掉的问题（都在 M4 工具里，引擎没改）

| # | 问题 | 后果 | 修法（`B-test\d3lib.py`） |
|---|---|---|---|
| 1 | M3a 的第四行打的是 `restage N (M guesses)`，末尾还多了 `\| mode B0 slots 3`；M4 的 `RE_L4` 要的是 `restage N \| chain waits` | 第四行一条都解析不出来：`trace_d3.py` 的 "B0: restage ≤ 5%" 判据和 held1/held2 对照永远不出现；`decode_timing_gain.py` 拿不到实测 ρ。`ab_d3.py` 判断"新循环跑了没有"用的是子串，不受影响 | 正则里加可选的 `(?: \(\d+ guesses\))?`；末尾多出来的字段本来就不影响 `search` |
| 2 | M3a 在每个请求头 `# request p0=N` 后面多写一行 `# mode=B0`；`read_trace` 把所有 `#` 开头的行都当成新请求 | 不带 marks 文件时，每个请求拆成一个空请求加一个 p0=−1 的请求，名字错，按 p0 分组也错。`ab_d3.py` 带 marks 文件时，事件会被拼回去，所以不受影响 | 没有 `p0=` 的 `#` 行直接跳过 |

- 核对：
  - `B-INT-work\check_parsers.py` 从 `generate.cpp` 里取出 `fprintf` 的格式串，重建第四行、状态行和 trace 头，用 d3lib 去解析。
  - 修之前两项都 FAIL，修之后 PASS（`check_parsers_out.txt`）。
  - 回归：`trace_d3.py` 跑 10-17 trace，输出和原来的 `trace_d3_1017_out.txt` 逐字相同；`decode_timing_gain.py --requests` 跑现有日志，除了原文件多出来的 `--sim` 两行，其余逐字相同。
  - 修改前的 d3lib 留了一份 `B-INT-work\d3lib.py.orig`。
- **新增** `B-test\compare_timing.py`：把一次 `ab_d3.py` 运行里两个臂的 timing 第 1、2 行逐请求对比。比的是窗口数、猜测数、猜中数、回滚数、门槛挡下数、新/猜测窗口数、强制链不一致数、链晚数，再加 ms 的偏差。N2 等价性检查用它。

## 4. 给调用方的 8190 命令

前提：
- 被测 exe 是 `builds\strata-p1b-build\strata.exe`（sha256 前 16 位 `93857b6ae1972fc7`）。`ab_d3.py` 开跑时会把它复制进 OUTDIR，之后再编译不影响这次运行。
- `ab_d3.py` 占用 PH402 的四颗 die（`-Devices 2,3,4,5`，也就是 8111 那四颗），端口 8190。
  - 它不启停 8110/8111：每 60 秒查一次 die 是否空闲，最多等 60 分钟，等不到就报 "GPU not available" 退出。
  - 8111 要不要先停，由用户决定。
- 每个臂各启动一次，用临时复制的 SwiftAbl 启动脚本。状态目录是 `OUTDIR\state`，不写 8111 的 runtime-state。
- 每个臂都开 `STRATA_DECODE_TIMING=1`、`STRATA_PIPELINE_DEBUG=1`、`STRATA_PIPELINE_TRACE`、`STRATA_PIPELINE_CHECK_STATE=1`。
- 在 `F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-test` 下运行（PowerShell）。

### 4.1 N2 等价性（检查翻译层；约 35–40 分钟）

```powershell
python -I F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-test\ab_d3.py F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-G2-n2 --arms S,P,N2,P0
python -I F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-test\compare_timing.py F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-G2-n2 --ref P --arms N2,P0
```

- 臂：
  - S：p1-b，pw 0，串行参考；
  - P：p1-b，pw 2，旧循环；
  - N2：p1-b，pw 2 + `STRATA_PIPELINE_NEWLOOP=1`，新循环跑旧语义；
  - P0：生产 `engine-sm60\strata.exe`，pw 2。
- **硬条件**：
  - `summary.txt` 里回复哈希 S = P = N2 = P0，8/8，并且和 mtp-v2 的参考哈希相同（"same"）；
  - 状态摘要 P、N2 和 S 一致（同一个 `end` 时）。
- **timing 行等于 P 臂**（`compare_timing.py`）：
  - 第 1、2 行的各项计数，N2 对 P 应当全部相同；
  - 个别计数（链晚、某个猜测窗口因为链晚而没发）两次运行之间本来就会动，以 P0 对 P 的那一行作为噪声底，N2 的不同不应多于它；
  - 解码循环的 ms 合计在 ±1% 以内。
- **trace**：`trace_d3.txt` 里 "N2: verdict sequence (a, on) equal to P in k/k requests" 和 "N2: trace tok/s within ±1%" 两条应为 PASS。
- **启动日志**：
  - N2 和 P 都应当是今天那行 `--pipeline-windows 2: 216 MiB of CUDA0 kept out of the expert cache (the second verifier, the GDN snapshots)`，没有 "slot pool" 行；
  - N2 臂热身后有 `strata pipeline depth3: … mode N2 slots 2`，P 臂没有这一行。`ab_d3.py` 会自动报 WARNING。
- **不过时**：哈希不同就是翻译层有错，停在这里，不跑 4.2。
  - 先看第一个不同的裁决（`trace_d3.txt` 里的 "first different verdict #k"）；
  - 再看 `STRATA_PIPELINE_LOG=1` 下的 cases 行。

### 4.2 B0 对 P（按段撤销 + 3 个槽；约 30–35 分钟）

```powershell
python -I F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-test\ab_d3.py F:\AI-Speed-Optimization\analysis\2026-10-19-p1-stack\B-G2-b0 --arms S,P,B0
```

- B0 臂：p1-b，pw 2 + `STRATA_PIPELINE_UNDO_PER_STAGE=1`。启动时 pl_ns = 3。
- **启动日志**：
  - `--pipeline-windows 2: 404 MiB of CUDA0 kept out of the expert cache (3 slots: 2 more verifiers, the GDN snapshots)`，即 2 × 160 + 3 × 28；
  - `the slot pool holds 3 slots`；
  - 仍然是 `100% VRAM resident`。
  - 如果出现 `the slot pool stays at two slots: …`，说明额外槽没分配成功，B0 会退回旧循环，这一臂的数就作废。
- **硬条件**：回复哈希 S = P = B0（8/8）；状态摘要 B0 = S。
- **判据**（设计第 9 节第 3 步，`summary.txt` 的 G2 criteria 会自动打勾）：
  - B0 对 P 按词加权 ≥ +1.0%，预期 +1.7%；
  - `trace_d3.txt` 里 B0 合并的 F 比 P 低 ≥ 0.7 ms，模拟器预计约 −1.1；
  - `restage` ≤ 新窗口数的 5%。这一条要第 3 节修好的正则才会出现。
- 也可以一次跑完：`ab_d3.py <OUTDIR> --stage g2` 就是 S、P、N2、B0、P0 五个臂，约 50 分钟。分开跑的好处是 N2 不过就不用再花 B0 的时间。

### 4.3 现在不要跑的

- `--stage g3`（X 臂 = `--pipeline-windows 3`）：M3a 里 pw 3 打印 "depth 3 is not wired yet" 后按 B0 跑，X 臂量到的就是 B0。要等 M3b 接上深度 3、11 输出链和 `chain_stop`。
- `stress_d3.py`：它的配置一半是 X。B0 部分现在可以跑，但设计把它排在 G3 之后。

## 5. 给 M3b 的提醒（M3a 第 8 节之外，这一轮看到的）

- 第四行统计的格式改了，就要同步改 `d3lib.RE_L4`。`check_parsers.py` 会直接从 `generate.cpp` 取格式串，用它核对。
- trace 头如果再加 `#` 行，不要带 `p0=`，否则会被当成新请求。
- `ab_d3.py` 的 X 臂只设 `AB_PW=3`。M3b 如果把 `STRATA_MTP_CHAIN_STOP` / `CHAIN_MAX` 改成要 DEBUG 才读，`d3route.PIPE_KEYS` 已经包含这两个，环境变量里加上即可。

## 6. 文件

- `B-INTEGRATION.md`：本文件。
- `B-INT-build.log`：构建日志。
- `B-INT-work\`：
  - `pl_sched_test_full_out.txt`、`pl_sched_test_time.txt`；
  - `parity_seed1_d4.log`、`parity_seed7_d5.log`；
  - `generate_daeed844.cpp`、`check_oldloop.py`、`check_oldloop_out.txt`；
  - `check_parsers.py`、`check_parsers_out.txt`；
  - `trace_d3_1017_recheck.txt`、`decode_timing_gain_recheck.txt`；
  - `d3lib.py.orig`。
- `B-test\d3lib.py`：已修改（第 3 节）。
- `B-test\compare_timing.py`：新增。
- 提交：`p1-b` 上的 `42a0734a`，只加本文件的副本 `docs/P1_B_INTEGRATION.md`（提交时的版本，不含这个哈希）。引擎代码没有改动可提交；M4 工具不在引擎树里。

## 7. 修复轮（Fix pass）—— 2026-10-19 晚

依据：`B-review-identity.md` 第 2.1、2.2 节，`B-review-engineering.md` 第 1、2、3 节。五条 major 全部处理。

- GPU 只用了板 B 的 die 4（19:00）和 die 5（1A:00），跑之前 `nvidia-smi --query-compute-apps` 查过：只有 01:00、02:00（3090 Ti）上有别的进程，两颗都空着，没有等。
- 路由一个没碰；生产文件（`engine-sm60`、`scripts\`、`runtime-state`）没碰。
- 旧的 pw-2 循环仍然逐字节不变（第 7.3 节）。不设新开关时的行为也不变：改动全在新循环、槽位池（pl_ns > 2）和 DEBUG 开关后面。

### 7.0 结论

| # | 复核意见 | 改在哪 | 结果 |
|---|---|---|---|
| 1 | 状态比较不看 `end`：被 max_new 截断的最后一个窗口提交自己的 a+1，S 臂和流水线各臂切窗不同，`end` 不同时四个哈希必然不同，P、N2、B0 对 S 会大量误报 "!"（identity 2.1、engineering 1） | M4：`d3lib.py`、`ab_d3.py`、`stress_d3.py` | `RE_STATE` 取 `end`、`loop`；只在同一个 `end` 时比较，不同 `end` 记 "~"（不可比），单独计数；N2、B0 对 P 比，P 对 S 比；硬条件改成"可比的全部相同，且每臂可比的 ≥ `--min-state`（默认 3）"；加了一个以 EOS 结尾的请求和一次停放后续写。合成数据核对 PASS |
| 2 | `chain_n` 语义不一致：引擎的 `chain_n()` 是"已排进的输出数"，调度器当成钳位后的 n_out，开了 `chain_stop` 的 mode B 会卡死；mock 按另一种语义，测不出（identity 2.2） | M1：`MtpDrafter::chain_cap()`；M2：`PlIo::chain_cap()`、`launch_chain`、`chain_round`、mock | 调度器发链后读 `chain_cap()`，链结束（r == 1）时用 `chain_n()` 重新取实际排了多少。mock 改成引擎语义：**把调度器临时改回旧读法，(c) 穷举 mode B 有 42,698 / 57,600 个世界报 deadlock；改回后 0**。全量 PASS |
| 3 | 额外槽的图在第一个流水线请求时才捕获，失败就 `return 1`；shrink 之后预留不退回、也不说（engineering 2） | M3：`pl_prepare`、`pl_pool_shrink` | 先捕获槽 0、1（失败照旧报错），再捕获槽 ≥ 2；失败就释放这些槽的验证器、hand-off、快照，池子退回 2 个槽，说一次原因。shrink 时打一行"CUDA0 / 其他段各有多少 MiB 白留到重启"。新加 DEBUG 测试开关 `STRATA_PIPELINE_POOL_FAIL=capture` |
| 4 | 选模式只看 `pl_ns < 3`，不按深度；`begin` 失败就 `die` 退出整个 serve（engineering 3） | M2：`Sched::check`；M3：选模式、新循环入口 | 按深度选：B 要 4 个槽，B0 要 3 个，N2 要 2 个，不够就用旧循环并说一次原因。调度器配置在排任何操作之前先过 `Sched::check`，过不了就说一次原因、这个请求走旧循环 |
| 5 | `die` 之前不打调度器状态（engineering 3 后半） | M2：`Sched::dump`；M3：`die` | 新循环出错时先打一行调度器状态：mode、ns、end、nwin、ndoom、撤销（seq、slot、keep、pending）、每个窗口和 doomed 的 seq/uid/slot/T/p/fr/fl/com、held、链的状态、各槽 owner（`*` = 源已发链）。新循环结束和池子 shrink 时，看门狗表的槽 2..4 清空（engineering 5 的 minor 顺手做了） |

### 7.1 引擎提交（p1-b，一个模块一个，没有署名行）

| 提交 | 模块 | 文件 | 内容 |
|---|---|---|---|
| `7efe310d` | M1 | `mtp.hpp`、`mtp_chain_parity.cpp` | `chain_cap()` 访问器（= n_out 钳位后，`chain_extend` 的上限）；`chain_n()` 的注释写明"到现在为止排进的数"。对拍程序在开着的链发出时核对 `chain_cap() == 11`、`chain_n() == min(nq, 11)`，在上下文边核对两者都等于钳位后的数 |
| `cda5afcb` | M2 | `pl_sched.hpp/.cpp`、`pl_sched_test.cpp` | `PlIo::chain_cap()`；`launch_chain` 用它定 `ch_.n`；`chain_round` 在 r == 1 时取 `min(ch_.n, io.chain_n())`。`Sched::check(Config)`（`begin` 先调它，错误信息带模式要几个槽、有几个）；`Sched::dump()`。mock：`chain_cap()` 返回钳位后的 n，`chain_n()` 按 `Timing::chain_n_engine` 返回"已排进的数"（默认，引擎语义）或旧读法；随机世界里 1/4 用旧读法（由种子另算，不动原来的随机流，所以世界和联调时相同） |
| `5671ad9f` | M3 | `generate.cpp` | `f_chain_cap`；选模式按深度；调度器配置挪到新循环入口之前、`Sched::check` 不过就置 `pl_want` 走旧循环（`pl_want` 去掉 `const`，旧循环那一块没动）；`CHECK_STATE` 的 `loop` 标签按实际跑的循环（`pl_go`）；`pl_prepare` 分两段捕获、槽 ≥ 2 失败就 shrink；额外快照的释放抽成 `pl_snaps_drop_x`（启动时的失败路径也用它）；`pl_pool_shrink` 打白留的 MiB、清看门狗表、stage 0 的验证器在自己的设备上释放；`die` 打 `Sched::dump()` |

- 每请求的退回路径（都只说一次）：
  - 启动时池子就小：`--pipeline-windows 3` 但 NS < 4 → "depth 3 needs 4 slots …; today's loop"（M3b 删掉 B→B0 降级之后才会走到；现在 pw 3 仍按 B0 跑）；B0 但 NS < 3 → "the per-stage undo needs 3 slots …"。
  - 第一次 `pl_prepare` 时额外槽捕获失败：启动后的第一个流水线请求打 "the slot pool stays at two slots: slot q's graphs on stage s: …" 和 "N MiB of CUDA0 and M MiB of every later stage's card stay kept out of the expert caches for the k slots it does not hold (until a restart)"。这个请求打 "the new pipelined loop cannot run this request (pl_sched: slots out of range (mode B0 needs 3, has 2)); today's loop"，之后的请求打 "the per-stage undo needs 3 slots …"。都按旧循环跑，回复不变。
- 白留的显存（池子从 NS 退到 2 时）：CUDA0 = 160 × (NS − 2) MiB + (NS − 2) 份 GDN 快照；其他每颗 die = 160 × (NS − 2) MiB。B0（NS 3）约 188 / 160 MiB，B（NS 4）约 376 / 320 MiB。

### 7.2 M4 工具（不在引擎树里，`B-test\` 下原地改，改前的副本在 `B-FIX-work\*.orig`）

- `d3lib.py`：
  - `RE_STATE` 加可选的 `end (-?\d+) loop (\S+)`（旧引擎不打也能解析）。
  - `state_digest` 对 (st, gdn, kv, idx, ple, end) 求摘要，**不含 `loop`**（各臂的循环名本来就不同）。
  - 新增 `state_end`、`state_loop`、`state_cmp(r, ref)`：`=` 同 end 相同，`!` 同 end 不同，`~` end 不同（不可比），空 = 没得比。
  - 道理：同一个 `end` 时两臂提交的 token 序列一样（都是串行的输出的前缀），掩码下的状态必须逐位相同；`end` 不同就什么也说明不了。
- `ab_d3.py`：
  - 每个请求记 `state_end`、`state_loop`。
  - 比较对象：N2、B0、X（以及 B0F）对 P（窗口逻辑相同，`end` 多半相同）；P 对 S；新循环各臂另外对 S 在同 `end` 时再比一次。`!` 列出请求名、臂、`end`。
  - summary 每臂一行：相同 / 不同 / 不可比的条数。G2、G3 的状态判据改成"同 end 的全部相同，且每臂可比的 ≥ `--min-state`（默认 3）"；可比的不够时写 "too few comparable requests"，不当成失败也不打勾。
  - 六条短请求之后加 3 个请求：
    - `eos_zh`："只回答一个字：好。"，512 token，以 EOS 结束，各臂提交的 token 数一样，状态能和 S 比；
    - `park_other`：另一段会话（`eos_zh` 那段被停放）；
    - `cont_eos_zh`：回到 `eos_zh` 那段会话（助手轮用 S 臂的回复），续写 300 token。它的回复哈希依赖 `eos_zh` 解码留下的状态，这就是设计第 9 节第 5 步的"停放 → 恢复 → 续写"，提前放进 G2。看它 serve 行的 `reused`：如果只复用到第一轮提示，说明会话缓存没带上解码后的状态，这一条就退化成普通的回复比对。
  - 新的可选臂 `B0F`（只在 `--arms` 里点名时跑）：B0 + `STRATA_PIPELINE_POOL_FAIL=capture`。热身后记录有没有 "slot pool stays at two slots"、白留 MiB 那一行、退回旧循环的那一行；不应有第四行 timing；回复要 = S，状态对 P 比。
- `stress_d3.py`：每个请求记 `state_end`；状态先和 S 比，`end` 不同就和本 case 里第一个同 `end` 的配置比，都没有就记 `~`。summary 报"同 end 比了多少、不可比多少"，只有同 end 不同才算失败。
- `d3route.py`：`PIPE_KEYS` 加 `STRATA_PIPELINE_POOL_FAIL`，换臂时清掉，不会漏进下一臂。

### 7.3 复跑的测试

| 测试 | 结果 | 输出 |
|---|---|---|
| 构建 `strata pl_sched_test mtp_chain_parity` | `BUILD_OK`。`generate.cpp` 新增行只有全树都有的 C4996；`pl_sched*.cpp` 在 `/W4` 下 0 警告。`strata.exe` sha256 前 16 位 **`83bcd8d38ca73e69`**，`--help` 能跑 | `B-FIX-build.log` |
| `pl_sched_test` 全量 | **PASS，0 失败**，425 s（和对拍同时跑）。输出除了耗时，和联调那次（`B-INT-work\pl_sched_test_full_out.txt`）**逐字相同**：调度器在引擎语义下的行为和原来在旧 mock 下一样。(d) N2 对旧循环 802,830 个世界逐行相同 | `B-FIX-work\pl_sched_test_full_out.txt` |
| `pl_sched_test --quick` | PASS，20.6 s | `B-FIX-work\pl_sched_test_quick_out.txt` |
| mock 能不能抓到旧问题 | 把 `launch_chain` 临时改回 `ch_.n = io.chain_n()`：`--quick --only c` 里 mode B **42,698 / 57,600 FAIL**，第一条是 "deadlock: the loop waits and nothing can happen on the GPU"；N2、B0 不受影响。改回后重编、全量 PASS | `B-FIX-work\pl_sched_test_oldread_c_out.txt` |
| `mtp_chain_parity`（板 B） | die 4 种子 1 × 300 轮、die 5 种子 7 × 400 轮都 **PASS**，七项计数和联调时完全相同，新加的 `chain_cap` 检查都过 | `B-FIX-work\parity_seed1_d4.log`、`parity_seed7_d5.log` |
| 旧 pw-2 循环逐字节 | daeed844 第 10482–11113 行（41,376 字节）在新文件里出现 **1 次**，没有 diff 操作落在里面 | `B-FIX-work\check_oldloop_out.txt` |
| 解析器对引擎格式 | `check_parsers.py` PASS（状态行带 `end 812 loop B0`） | `B-FIX-work\check_parsers_out.txt` |
| 状态比较工具 | `B-FIX-work\check_state_tools.py`（新）：从 `generate.cpp` 的格式串重建状态行；`end`/`loop` 解析、摘要不含 `loop`、`~`/`!`/`=` 三种结果；合成的 G2 运行（S 在 7/11 个请求上 `end` 不同）不再误报、状态判据打勾；同 end 改一个 GDN 哈希就报 "different: code_en: B0 vs P"、判据不打勾；`stress_d3` 合成矩阵不同 end 不算失败。14 项全 PASS | `B-FIX-work\check_state_tools_out.txt` |
| 回归 | `trace_d3.py` 跑 10-17 trace（带 `--log`），输出和 `B-test\trace_d3_1017_out.txt` 逐字相同 | `B-FIX-work\trace_d3_1017_recheck.txt` |

- 真机冒烟：仍然没有。单颗 die 装不下模型；两条退回路径（池子 shrink、按深度选模式）只能在 8190 上看（第 7.4 节的 `B0F` 臂）。

### 7.4 对第 4 节命令的影响

- 命令不变。每臂多 3 个请求，约多 1–2 分钟。
- G2 判据改为：
  - 回复哈希 S = P = N2 = B0 = P0，**11/11**（8 条原请求 + `eos_zh` + `park_other` + `cont_eos_zh`）；mtp-v2 参考哈希只有原来 8 条有。
  - 状态：N2、B0 对 P，P 对 S，同 `end` 的全部相同，每臂可比的 ≥ 3。表里 `~` 是"end 不同、不可比"，不是失败。
  - 其余（N2 ±1%、B0 ≥ +1.0%、F 降 ≥ 0.7 ms、`restage` ≤ 5%）不变。
- 可选：加 `--arms S,P,B0F` 单跑一次退回路径（约 15 分钟）。看 `results.jsonl` 里 B0F 热身那一行的 `pool_fallback`、`reserved_note`、`fallback_note` 都是 true，回复哈希 = S。

### 7.5 给 M3b 的（这一轮之后还剩的）

- M3a 第 8 节第 1 条（`chain_cap`）已做；第 2 条 `set_chain_max(11)`（identity 2.3）、第 3 条删 B→B0 降级、读 `STRATA_MTP_CHAIN_STOP`/`CHAIN_MAX` 仍待 M3b。删降级后，pw 3 但 NS < 4 的引擎会走"depth 3 needs 4 slots"那条退回，而不是退出。
- 调度器其他不变量断言失败仍然 `die`（现在会先打 `Sched::dump()`）。engineering 第 4 节的池子退回测试钩子只做了 `capture` 一种；`init`、`snap` 两种和 io 错误注入（engineering 6.2）留给 M3b。
- engineering 的 minor 没在这一轮做：`stage_room` 后段的快照预留、验证器登记 16 个的提示、`static_assert`、`pl_sched_test` 的 CMake 位置、`CHECK_STATE` 的逐格拷贝慢。
