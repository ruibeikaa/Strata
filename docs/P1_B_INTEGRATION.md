# 部分 B：M1–M4 联调（深度 2：N2 / B0）—— 2026-10-19

工作树 `builds\strata-p1b`（分支 `p1-b`，HEAD `b227571e` = daeed844 + M1 `11ec28f7` + M2 `caf0612b` + M3a `b227571e`）。
- 引擎代码这一轮**一行没改**：联调里查出来的两处不一致都在 M4 的 Python 工具里，已修。
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
- 提交：`p1-b` 上一个提交，只加本文件的副本 `docs/P1_B_INTEGRATION.md`。引擎代码没有改动可提交；M4 工具不在引擎树里。
