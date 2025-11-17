# RunMorphCast 情感与生理数据分析工具

一个可在浏览器本地运行的情绪/生理数据分析工作台。前端使用纯 HTML/JavaScript 构建，后端通过 Flask + DeepFace + PyOD + NeuroKit2 提供分析服务（默认以 Docker 方式运行）。

---

## 🚀 快速上手

| 步骤 | 操作 |
| --- | --- |
| 1 | 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| 2 | 启动后端：`cd deepface-api && docker compose up --build -d` |
| 3 | 打开 `MainPage.html`（建议使用 VS Code Live Server 或任意本地静态服务器） |
| 4 | 在页面中导入视频/生理数据后开始分析 |

---

## 📁 项目结构

```
RunMorphCast/
├── MainPage.html                # 前端主页面（本地打开即可）
├── README.md                    # 当前文档
├── PROJECT_ANALYSIS_AND_OPTIMIZATION.md  # 项目分析/重构方案
└── deepface-api/                # 后端服务
    ├── deepface_api.py          # Flask API（DeepFace / PyOD / NeuroKit2）
    ├── requirements.txt         # 后端依赖
    ├── Dockerfile               # Docker 构建配置
    └── docker-compose.yml       # Docker Compose 启动配置
```

---

## 🧩 功能概览

| 功能 | 描述 |
| --- | --- |
| 视频情绪分析 | 使用 MorphCast/DeepFace 获取 Valence/Arousal + 七类情绪 |
| 生理信号处理 | 支持 GSR/PPG 等原始信号导入、清洗、导出 |
| 异常检测 | 通过 PyOD Isolation Forest 检测异常区段 |
| 数据导出 | 支持 CSV/XDF 等格式 |

---

## ⚙️ 后端部署

### 方式一：Docker（推荐）

```bash
cd deepface-api
docker compose up --build -d   # 首次运行
docker compose logs -f         # 查看日志
docker compose down            # 停止
```

服务启动后可访问：
- http://localhost:5000/health
- http://localhost:5000/anomaly/info
- http://localhost:5000/eda/info

### 方式二：Python 本地运行（如不使用 Docker）

```bash
cd deepface-api
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python deepface_api.py
```

---

## 🖥️ 前端使用

1. 使用 VS Code 打开项目
2. 启动 Live Server（或 `npx serve .` 等静态服务器）
3. 访问 `http://localhost:5500/MainPage.html`（端口取决于你的静态服务器）
4. 在页面顶部检查“后端 API 状态”，确认连接成功
5. 即可上传视频/生理数据，进行分析与导出

> 若直接双击打开 `MainPage.html`，部分浏览器会阻止本地文件访问，建议使用静态服务器以避免 CORS/跨域问题。

---

## 🔧 常见问题

| 问题 | 解决方案 |
| --- | --- |
| 访问 `http://localhost:5000` 失败 | 检查 Docker 是否运行；`docker compose logs` 查看错误；确认 5000 端口未被占用 |
| CORS 报错 | 已在 `deepface_api.py` 中启用 `flask-cors`；若仍报错，确认前端访问的端口与后端一致 |
| 性能较慢 | 减少视频帧采样率；关闭无用的数据流；确保只在需要时开启异常检测 |
| Docker 镜像体积大 | 第一次构建会下载模型/依赖，属于正常行为 |

---

## 📚 参考资料

- `PROJECT_ANALYSIS_AND_OPTIMIZATION.md`：现状分析与重构方案
- `deepface-api/README_EDA_Docker.md`：后端 API 详细说明
- MorphCast / DeepFace / PyOD / NeuroKit2 官方文档

---

如需协助或希望贡献代码，欢迎提交 Issue 或 PR。祝使用愉快！
