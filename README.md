# Lensplit

本地离线运行的 Windows 视频分镜工具：把一条素材自动切成镜头，逐个镜头标注人数、分配演员、核对档期与拍摄日，最后按镜号导出成片。

**素材、模型、台词与演员信息全部只在本机处理，不联网、不上传。**

---

## 功能

### 页面 1 · 剪辑

- **AI 镜头切分**：内置 TransNetV2（ONNX），阈值可调；模型缺失时仍可用「快速模式」（画面差异检测）。
- **手动切点**：进度条限定在**当前所选镜头**内，拖动精度落在这一段的几秒里；点「标记切点」把当前帧记成切点，可一次应用全部标记。轨道上蓝线＝已有切点，黄线＝待应用标记。
- **镜头合并 / 拆分 / 撤销**，以及镜号编排（样式、起始值、位数）。
- **人物自动标记**：内置 YOLOv8n（ONNX），按镜头抽样多帧取人数最多的一帧，给出「本镜人数」。
- **角色槽**：人数 = N 时自动生成「角色 1…角色 N」，每个槽都能改名，并直接从演员库下拉选人；槽后显示演员的**特点备注**。填了本镜**拍摄日期**后，与演员档期冲突的角色当场标红「档期冲突」。
- **台词识别**：内置 whisper-small（CTranslate2），逐镜转写，可手动修正。

### 页面 2 · 拍摄表

按镜头汇总：镜号、时长、人数、角色与演员、拍摄日期、台词，支持导出 CSV。

### 页面 3 · 演员与档期

- 演员库增删改：姓名、可拍摄开始 / 结束、备注（特点、造型要求等）。
- **档期看板**：演员名 + 起止档期 + 参演镜头数 + 占用条 + 状态。
- **CSV 批量导入**，同名视为更新，不会重复建人。

---

## 环境要求

| 项目 | 说明 |
| --- | --- |
| 系统 | Windows 10 / 11（界面基于 PySide6，其他平台未验证） |
| Python | 3.10 及以上 |
| ffmpeg | 导出镜头需要，`ffmpeg` + `ffprobe` 放在 `PATH`，或用环境变量 `LENSPLIT_FFMPEG` / `FFMPEG_PATH` / `FFMPEG` 指定 |

## 安装

```bash
git clone https://github.com/ztdd2018/lensplit.git
cd lensplit
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 模型准备

| 模型 | 用途 | 获取方式 |
| --- | --- | --- |
| `models/transnetv2.onnx` | 镜头切分 | 随仓库提供 |
| `models/yolov8n.onnx` | 人物计数 | 随仓库提供 |
| `models/whisper-small/` | 台词识别 | 运行下方脚本下载（约 484MB） |

```bash
python scripts/download_models.py          # 下载到 models/whisper-small
python scripts/download_models.py --force  # 已存在时强制重新下载
```

脚本从 HuggingFace 的 `Systran/faster-whisper-small`（`openai/whisper-small` 的 CTranslate2 转换）拉取权重。只保留 `config.json`、`tokenizer.json`、`vocabulary.txt` 在仓库里，避免仓库被 484MB 二进制撑大。

> 默认使用 CPU 版 `onnxruntime`。要启用 GPU，把基础依赖换成 `pip install -e ".[cuda]"`（NVIDIA）或 `pip install -e ".[dml]"`（DirectML），两者不能与 CPU 版共存。

## 启动

```bash
lensplit                      # 安装后的命令行入口
python -m lensplit            # 或在仓库根目录直接运行
```

Windows 下也可以双击仓库根目录的 `启动 Lensplit.bat`。

> 请在仓库根目录启动（或使用 `lensplit` 命令），程序会在当前目录与安装目录下寻找 `models/`。

---

## 使用流程

1. **导入素材**，选择检测引擎后点「开始分析」，得到镜头序列。
2. 在镜头序列里点中一个镜头 → 进度条自动缩放到该片段 → 拖动定位、必要时「标记切点」，然后应用。
3. 框选相邻镜头可**合并**，或对合并结果**拆分**；用镜号设置统一编号。
4. 页面 3 先建演员库（可 CSV 批量导入），填好档期与备注。
5. 回到页面 1，改「人数」得到 N 个角色槽 → 角色改名 → 从下拉框选演员 → 填本镜**拍摄日期**；档期不符的角色会立刻标红。
6. 需要时跑台词识别，并在时间线上修正文字。
7. `Ctrl+S` **保存工程**；下次 `Ctrl+O` 打开继续。
8. 「导出」按镜号输出视频，同时生成 CSV / JSON 清单。

### 演员库 CSV 格式

```csv
姓名,可拍摄开始,可拍摄结束,备注
张三,2026-11-01,2026-11-05,会骑马 古装造型
李四,,,新人 可长期跟组
```

- 第一行必须是表头；`姓名` 必填，**同名视为更新**（补填档期 / 备注），不会新建重复演员。
- 日期格式 `YYYY-MM-DD`，留空表示「不限」；开始日期不能晚于结束日期。
- 编码支持 UTF-8（含 BOM）与 GBK；单行出错只跳过该行并在结果里汇总。
- 页面 3 的「下载模板」可直接导出一份带示例的 CSV。

### 工程文件

`Ctrl+S` 会把整份工作保存成 `.lensplit`（JSON）：视频路径、镜头切分与合并关系、手动切点、台词、演员库、角色槽与演员分配、人数、拍摄日期、检测参数、编号设置。首次保存默认落在视频同目录。视频被移动后工程仍可打开查看与编辑，只是无法预览与导出。

### 导出

- **精确切点（重编码）**：按 PTS 精确切，自动探测 NVENC → QSV → AMF → CPU。
- **快速复制**：不重编码，切点会对齐到最近关键帧。
- 同时输出 UTF-8 CSV 与 JSON 清单，便于交给剪辑软件。

---

## 目录结构

```
src/lensplit/
├── core/       镜头模型、编号、演员与档期（casting）、工程读写（session）
├── analysis/   TransNetV2、YOLOv8n、whisper 与线程化 worker
├── video/      ffmpeg 探测、解码、缩略图
├── export/     镜头导出与清单
└── ui/         主窗口、时间线、拍摄表、演员与档期
models/         transnetv2.onnx、yolov8n.onnx、whisper-small/
scripts/        download_models.py
tests/          单元测试
```

## 测试

```bash
python -m unittest discover -s tests -q      # 120 项，含 UI（offscreen）
```

Linux / CI 上需先设置 `QT_QPA_PLATFORM=offscreen`。

## 已知限制

- 人物计数基于 YOLOv8n 检测当前帧里**可辨识**的人。远景小目标、背身、遮挡、人群叠压仍会漏检；可通过提高「采样帧数」、压低「置信度」改善，或换用更大的 YOLO 权重（放到 `models/yolov8n.onnx` 同路径即可）。
- 台词识别为 whisper-small，长素材首次加载模型需要数秒。
- 目前只在 Windows 上验证过。

## 隐私

没有网络请求、没有遥测、没有云同步。视频、模型推理、台词文本、演员资料都留在本机，工程文件也只有你自己拷走才会移动。
