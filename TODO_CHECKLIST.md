# self-evolution-Agent 待完成清单

本文档用于跟踪 `self-evolution-Agent` 从当前代码状态到端到端可用、可部署、可验收所需完成的事项。

> 模型选型必须由项目所有者决定。文档中出现的模型名称只代表当前代码默认值或候选占位，不代表最终选型。

## 当前状态

- [x] FastAPI API 基础服务已实现
- [x] SQLite 持久任务队列已实现
- [x] LangGraph Planner 和专业 Agent 工作流已实现
- [x] LangGraph interrupt/resume 与 SQLite checkpoint 已实现
- [x] Inspiration、Fridge、Placeholder Agent 已实现
- [x] ChromaDB RAG 基础能力已实现
- [x] 飞书消息、图片、卡片和 Bitable 客户端已实现
- [x] 独立视觉服务代码已实现
- [x] QLoRA 数据导出、训练和评测脚本已实现
- [x] Python 虚拟环境已创建
- [x] 自动化测试已通过
- [x] 已检测到 NVIDIA RTX 4060 Laptop GPU 8GB
- [x] 创建项目 `.env`
- [ ] 完成全部模型选型
- [ ] 配置 Chat API 和密钥
- [ ] 完成飞书应用配置
- [ ] 完成飞书多维表格配置
- [ ] 配置公网 HTTPS 回调地址
- [ ] 安装并验证 Docker 环境，或确定使用纯 Python 部署
- [ ] 下载并验收最终视觉模型
- [ ] 完成两个业务闭环的端到端验收

---

## P0：模型选型

### Planner 模型

- [x] 确定 Planner 模型供应商
- [x] 确定 Planner 模型名称和版本
- [x] 确定使用云端 API 还是本地部署
- [ ] 确认支持中文意图识别和任务拆分
- [ ] 确认是否支持 JSON Schema Structured Output
- [ ] 确认上下文窗口满足多意图请求
- [ ] 确认 API 价格和调用限额
- [ ] 确认数据保留和隐私政策
- [x] 决定 Planner 是否与专业 Agent 共用模型
- [x] 填写 `PLANNER_MODEL`

### 专业 Agent 文本模型

- [x] 确定 Inspiration Agent 使用的模型
- [x] 确定 Fridge Agent 菜谱生成使用的模型
- [x] 确定库存动作参数提取使用的模型
- [x] 确定 RAG 答案生成使用的模型
- [x] 确认是否允许不同专业 Agent 使用不同模型
- [ ] 确认模型支持严格 JSON 输出
- [ ] 确认模型中文生成质量
- [ ] 确认模型 API 限流和超时策略
- [ ] 确认是否允许用户知识内容发送给模型供应商
- [x] 填写 `CHAT_MODEL`

### Embedding 模型

- [ ] 确定 Embedding 模型名称和版本
- [ ] 确定本地推理或远程 Embedding API
- [ ] 确认中文语义检索效果
- [ ] 确认向量维度
- [ ] 确认最大输入长度
- [ ] 确认是否需要查询指令前缀
- [ ] 确认模型许可证和商用范围
- [ ] 确认 CPU 推理速度和内存占用
- [ ] 准备一组真实知识检索评测样本
- [ ] 对候选模型运行相似度和召回率测试
- [ ] 填写 `EMBEDDING_MODEL`
- [ ] 如果选择远程 API，新增对应的 Embedding Provider

当前代码默认值，仅作为占位：

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 视觉/OCR 模型

- [ ] 确定视觉/OCR 模型名称和版本
- [ ] 确定本地模型或远程视觉 API
- [ ] 确定模型参数规模
- [ ] 确定量化格式：NF4、AWQ、GPTQ、GGUF 或其他
- [ ] 确定模型服务框架
- [ ] 确认 RTX 4060 8GB 是否能稳定运行
- [ ] 确认单张图片的最大分辨率和文件大小
- [ ] 确认模型并发限制
- [ ] 确认模型输出可严格匹配 `VisionResult`
- [ ] 确定食材名称置信度阈值
- [ ] 确定日期识别置信度阈值
- [ ] 确认模型许可证和使用范围
- [ ] 填写 `VISION_MODEL_NAME`
- [ ] 填写 `VISION_MODEL_VERSION`
- [ ] 填写或确认 `VISION_BASE_URL`

当前代码默认值，仅作为占位：

```env
VISION_MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
VISION_MODEL_VERSION=qwen2.5-vl-3b-4bit-v1
```

### 微调底座模型

- [ ] 确定 QLoRA 微调底座模型
- [ ] 确定模型版本是否与本地推理版本一致
- [ ] 确定云端训练平台
- [ ] 确定云端训练 GPU 类型
- [ ] 确定训练预算
- [ ] 确定 LoRA rank、alpha 和训练轮数
- [ ] 确定 Adapter 保存、合并和发布方式
- [ ] 确定微调模型量化方式
- [ ] 定义模型版本命名规则
- [ ] 定义模型回滚机制

### 可选 Reranker

- [ ] 决定第一版是否引入 Reranker
- [ ] 如果引入，确定本地模型或远程 API
- [ ] 确定初始向量召回数量
- [ ] 确定重排序后保留数量
- [ ] 确定可接受的额外延迟
- [ ] 准备检索质量对比评测

---

## P0：API 与凭据

### Chat API

- [x] 获取 OpenAI-compatible API Base URL
- [x] 获取 API Key
- [x] 获取 Planner 模型 ID
- [x] 获取专业 Agent 模型 ID
- [ ] 确认 API 账户余额或配额
- [ ] 确认请求频率限制
- [ ] 确认并发限制
- [ ] 确认请求超时建议
- [ ] 确认 Structured Output 兼容情况
- [ ] 确认日志和数据保留政策

待填写配置：

```env
CHAT_BASE_URL=
CHAT_API_KEY=
CHAT_MODEL=
PLANNER_BASE_URL=
PLANNER_API_KEY=
PLANNER_MODEL=
```

### Hugging Face 或模型仓库

- [ ] 创建模型仓库账号
- [ ] 获取 Access Token（如需要）
- [ ] 同意目标模型许可证（如需要）
- [ ] 确认本机或服务器能够访问模型仓库
- [ ] 确认模型缓存目录
- [ ] 检查磁盘剩余空间
- [ ] 确定是否使用镜像、离线下载或私有模型仓库
- [ ] 确定模型文件的备份方式

可选配置：

```env
HF_TOKEN=
HF_HOME=./models/huggingface
```

### 公网 HTTPS 入口

- [ ] 决定使用 Cloudflare Tunnel、ngrok、云服务器或其他方案
- [ ] 准备公网域名或隧道地址
- [ ] 确认公网入口支持 HTTPS
- [ ] 将公网请求转发至 API `8000` 端口
- [ ] 确认事件回调地址能够被飞书访问
- [ ] 确认卡片回调地址能够被飞书访问
- [ ] 配置访问日志
- [ ] 配置隧道或反向代理自动重启

最终回调地址：

```text
https://<公网域名>/webhooks/feishu/events
https://<公网域名>/webhooks/feishu/actions
```

---

## P0：飞书应用

### 创建应用和机器人

- [ ] 创建飞书企业自建应用
- [ ] 启用机器人能力
- [ ] 设置机器人名称
- [ ] 设置机器人头像
- [ ] 设置机器人描述
- [ ] 设置应用可用范围，仅包含授权用户
- [ ] 创建应用版本
- [ ] 发布应用版本
- [ ] 将 Bot 添加到当前飞书工作区
- [ ] 确认授权用户能与 Bot 私聊

### 获取应用配置

- [ ] 获取 `FEISHU_APP_ID`
- [ ] 获取 `FEISHU_APP_SECRET`
- [ ] 获取 `FEISHU_VERIFICATION_TOKEN`
- [ ] 获取或决定是否使用 `FEISHU_ENCRYPT_KEY`
- [ ] 获取授权用户的 `open_id`
- [ ] 填写 `FEISHU_ALLOWED_OPEN_ID`

待填写配置：

```env
FEISHU_APP_ID=
FEISHU_APP_SECRET=
FEISHU_VERIFICATION_TOKEN=
FEISHU_ENCRYPT_KEY=
FEISHU_ALLOWED_OPEN_ID=
```

### 飞书权限

- [ ] 申请接收私聊消息事件的权限
- [ ] 申请读取用户消息的权限
- [ ] 申请读取消息图片资源的权限
- [ ] 申请以应用身份发送消息的权限
- [ ] 申请发送交互卡片的权限
- [ ] 申请接收交互卡片回调的权限
- [ ] 申请读取多维表格字段的权限
- [ ] 申请新增多维表格记录的权限
- [ ] 发布包含新权限的应用版本
- [ ] 确认管理员已批准相关权限

### 事件订阅

- [ ] 配置事件回调地址
- [ ] 完成飞书 URL Verification
- [ ] 订阅 `im.message.receive_v1`
- [ ] 确认事件正文加密处于关闭状态
- [ ] 确认 verification token 校验成功
- [ ] 如果配置 Encrypt Key，确认请求签名校验成功
- [ ] 发送文本私聊测试事件
- [ ] 发送图片私聊测试事件
- [ ] 确认群聊事件不会进入业务流程
- [ ] 确认非授权用户被拒绝

### 交互卡片

- [ ] 配置交互卡片回调地址
- [ ] 验证确认按钮能够回调
- [ ] 验证取消按钮能够回调
- [ ] 验证卡片表单返回 `form_value`
- [ ] 验证食材名称可以修改
- [ ] 验证食材数量可以修改
- [ ] 验证到期日可以修改
- [ ] 验证填写 `unknown` 的行为
- [ ] 验证重复点击不会重复写入库存
- [ ] 验证过期动作会被拒绝

---

## P0：飞书多维表格

### 创建和授权

- [ ] 创建用于保存灵感和 TODO 的多维表格
- [ ] 创建目标数据表
- [ ] 获取 Bitable `app_token`
- [ ] 获取 Bitable `table_id`
- [ ] 将多维表格授权给飞书应用
- [ ] 确认应用能够读取字段定义
- [ ] 确认应用能够新增记录

待填写配置：

```env
FEISHU_BITABLE_APP_TOKEN=
FEISHU_BITABLE_TABLE_ID=
```

### 创建字段

- [ ] 创建 `标题`，推荐类型为单行文本
- [ ] 创建 `内容`，推荐类型为多行文本
- [ ] 创建 `类型`，推荐类型为文本或单选
- [ ] 创建 `标签`，当前代码按逗号分隔文本写入
- [ ] 创建 `状态`，推荐类型为文本或单选
- [ ] 创建 `来源消息`，推荐类型为文本
- [ ] 创建 `创建时间`，类型为日期时间
- [ ] 确认所有字段名称完全一致
- [ ] 启动 Worker，验证字段校验通过
- [ ] 通过 Bot 写入一条测试灵感
- [ ] 通过 Bot 写入一条测试 TODO

---

## P0：项目 `.env`

- [x] 从 `.env.example` 创建 `.env`
- [ ] 填写飞书应用配置
- [ ] 填写飞书授权用户 open_id
- [ ] 填写 Bitable 配置
- [x] 填写 Chat API 配置
- [x] 填写 Planner 模型配置
- [x] 填写专业 Agent 模型配置
- [ ] 填写最终 Embedding 模型配置
- [ ] 填写最终视觉模型配置
- [ ] 检查数据库路径
- [ ] 检查 ChromaDB 路径
- [ ] 检查视觉服务地址
- [ ] 确认 `.env` 已被 `.gitignore` 排除
- [ ] 确认未在日志或截图中暴露密钥

创建命令：

```powershell
Copy-Item .env.example .env
```

最低必填项：

```env
FEISHU_APP_ID=
FEISHU_APP_SECRET=
FEISHU_VERIFICATION_TOKEN=
FEISHU_ALLOWED_OPEN_ID=
FEISHU_BITABLE_APP_TOKEN=
FEISHU_BITABLE_TABLE_ID=

CHAT_BASE_URL=
CHAT_API_KEY=
CHAT_MODEL=
PLANNER_BASE_URL=
PLANNER_API_KEY=
PLANNER_MODEL=

EMBEDDING_MODEL=

VISION_BASE_URL=http://vision:8001
VISION_MODEL_NAME=
VISION_MODEL_VERSION=
```

---

## P0：本地运行环境

### 已确认硬件

- [x] GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- [x] 显存：8188 MiB
- [x] NVIDIA 驱动：560.94
- [x] Python 3.12 可用
- [x] 项目 `.venv` 已创建
- [ ] Docker CLI 可用
- [ ] Docker Compose 可用
- [ ] 容器可以访问 NVIDIA GPU

### 部署方式决策

- [ ] 决定使用纯 Python 本地部署
- [ ] 或决定使用 Docker Desktop + WSL2 部署
- [ ] 或决定使用云端 API/数据层加本地视觉服务
- [ ] 记录最终部署拓扑
- [ ] 记录数据保存位置
- [ ] 记录模型保存位置

### Docker 方案

- [ ] 安装 WSL2
- [ ] 安装 Docker Desktop
- [ ] 启用 WSL2 backend
- [ ] 确认 Docker CLI 在 PowerShell 可用
- [ ] 确认 `docker compose` 可用
- [ ] 安装或启用 NVIDIA Container Toolkit
- [ ] 验证 CUDA 容器能够执行 `nvidia-smi`
- [ ] 构建 API/Worker 镜像
- [ ] 构建 Vision 镜像
- [ ] 启动 Compose
- [ ] 检查 API、Worker、Vision 日志

验收命令：

```powershell
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker compose up --build
```

### 纯 Python 方案

- [ ] 安装最终模型需要的 CUDA PyTorch
- [ ] 安装项目 `vision` 可选依赖
- [ ] 验证 `torch.cuda.is_available()` 为 `True`
- [ ] 验证 BitsAndBytes 可用
- [ ] 下载最终视觉模型
- [ ] 验证视觉服务可以启动
- [ ] 验证视觉服务 readiness 为 `200`
- [ ] 验证主 Worker 能访问视觉服务

验收命令：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

---

## P0：启动与基础连通性

- [ ] 启动 FastAPI API
- [ ] 确认 `/health/live` 返回 `200`
- [ ] 填写配置后确认 `/health/ready` 返回 `200`
- [ ] 启动 Worker
- [ ] 确认 Worker 初始化 SQLite
- [ ] 确认 Worker 初始化 ChromaDB
- [ ] 确认 Worker 初始化 LangGraph checkpoint
- [ ] 确认 Worker 校验 Bitable 字段成功
- [ ] 启动 Vision 服务
- [ ] 确认 Vision `/health/live` 返回 `200`
- [ ] 确认 Vision `/health/ready` 返回 `200`
- [ ] 确认 Worker 可以访问 `VISION_BASE_URL`
- [ ] 确认公网 HTTPS 地址可以访问 API

---

## P1：业务规则确认

### 知识与 RAG

- [ ] 确定知识类和灵感类的判断标准
- [ ] 确定短文本是否默认作为灵感
- [ ] 确定长文本入库阈值
- [ ] 确定知识分块大小
- [ ] 确定分块重叠大小
- [ ] 确定 RAG 默认召回数量
- [ ] 确定时间过滤语义
- [ ] 确定网页允许抓取的范围
- [ ] 确定网页内容保留周期
- [ ] 确定知识删除和更新方式
- [ ] 确定引用展示格式

### 冰箱库存

- [ ] 确定支持的食材类别范围
- [ ] 确定数量和单位标准
- [ ] 确定生产日期格式
- [ ] 确定到期日期格式
- [ ] 确定“日期未知”的业务规则
- [ ] 确定临期默认天数
- [ ] 确定已过期食材如何展示
- [ ] 确定是否允许数量为小数
- [ ] 确定消耗是整项消耗还是支持部分扣减
- [ ] 确定删除和消耗的区别
- [ ] 确定库存图片的保留期限

### 菜谱

- [ ] 确定是否允许外购辅料
- [ ] 确定外购辅料最大数量
- [ ] 确定是否考虑人数
- [ ] 确定是否考虑忌口和过敏
- [ ] 确定是否考虑烹饪时间
- [ ] 确定是否提供营养信息
- [ ] 确定生成菜谱后是否提供“确认消耗”按钮

### 现有默认参数确认

- [ ] 确认最大图片为 10 MiB
- [ ] 确认网页请求超时为 10 秒
- [ ] 确认网页正文最大为 2 MiB
- [ ] 确认 RAG 默认召回 5 条
- [ ] 确认任务最大尝试 5 次
- [ ] 确认 Worker 轮询间隔为 1 秒
- [ ] 确认待确认操作有效期为 24 小时
- [ ] 确认 stale job 恢复阈值为 15 分钟

---

## P1：训练数据与微调

### 数据采集

- [ ] 确定目标食材类别
- [ ] 确定目标包装类型
- [ ] 收集正常光照图片
- [ ] 收集反光包装图片
- [ ] 收集模糊图片
- [ ] 收集倾斜拍摄图片
- [ ] 收集复杂背景图片
- [ ] 收集超市小票图片
- [ ] 收集不同日期格式样本
- [ ] 获取训练图片和小票的合法授权
- [ ] 对姓名、手机号、会员号等敏感信息脱敏

### 标注

- [ ] 定义统一标注规范
- [ ] 定义食材标准化名称表
- [ ] 定义数量单位表
- [ ] 定义日期无法识别时的标注方法
- [ ] 定义一图多食材的标注方法
- [ ] 定义置信度标注或评估方法
- [ ] 检查自动采集的用户修正样本
- [ ] 对训练样本进行人工抽检

### 数据切分

- [ ] 按拍摄来源划分训练集、验证集和测试集
- [ ] 避免同一包装近重复图片跨集合
- [ ] 统计各类场景占比
- [ ] 保留固定的最终测试集
- [ ] 建立数据集版本号

### 云端训练

- [ ] 创建云端训练实例
- [ ] 配置训练平台凭据
- [ ] 上传脱敏后的数据集
- [ ] 安装 `train` 和 `vision` 依赖
- [ ] 运行 QLoRA 基线训练
- [ ] 保存训练日志
- [ ] 保存 Adapter
- [ ] 运行验证集评测
- [ ] 运行最终测试集评测
- [ ] 合并或发布 Adapter
- [ ] 重新量化部署版本
- [ ] 将模型同步回本机

### 评测门槛

- [ ] 确定严格 JSON 成功率门槛
- [ ] 确定食材名称准确率门槛
- [ ] 确定生产日期准确率门槛
- [ ] 确定到期日期准确率门槛
- [ ] 确定完整样本准确率门槛
- [ ] 确定错误样本人工复查流程
- [ ] 确定模型发布和回滚标准

当前计划中的建议门槛，尚待确认：

- 严格 JSON 成功率：`>= 99%`
- 食材名称准确率：`>= 90%`

---

## P1：端到端验收

### 飞书入口

- [ ] URL Verification 成功
- [ ] 授权用户私聊文本可以入队
- [ ] 授权用户私聊图片可以入队
- [ ] 未授权用户请求被拒绝
- [ ] 群聊请求被拒绝
- [ ] 不支持的消息类型被安全忽略
- [ ] 重复事件不会创建重复任务

### 灵感闭环

- [ ] 发送灵感文本
- [ ] Planner 路由到 Inspiration Agent
- [ ] 标题和标签提取正确
- [ ] Bitable 新增记录成功
- [ ] 重复事件不会重复写 Bitable
- [ ] 飞书收到处理结果

### TODO 闭环

- [ ] 发送 TODO 文本
- [ ] 正确识别为 TODO
- [ ] Bitable 类型字段正确
- [ ] 状态默认为待处理
- [ ] 来源消息 ID 正确保存

### 知识入库闭环

- [ ] 发送长文本知识
- [ ] 知识完成清洗和分块
- [ ] Embedding 模型生成向量
- [ ] ChromaDB 写入成功
- [ ] 来源 message_id 正确保存
- [ ] 发送公开网页链接
- [ ] 网页正文抓取成功
- [ ] 私网和本地 URL 被拒绝
- [ ] 超大正文被拒绝
- [ ] 非 HTML/文本内容被拒绝

### 知识查询闭环

- [ ] 查询已入库知识
- [ ] 返回语义相关片段
- [ ] 返回答案包含来源编号
- [ ] 验证时间范围过滤
- [ ] 验证没有结果时的提示
- [ ] 验证 Chat 模型不可用时的降级结果

### 食材录入闭环

- [ ] 从飞书发送食材照片
- [ ] Worker 下载图片成功
- [ ] 图片文件大小校验成功
- [ ] Vision 服务返回合法 JSON
- [ ] 识别草稿保存成功
- [ ] 飞书显示确认卡片
- [ ] 可以修改食材名称
- [ ] 可以修改数量
- [ ] 可以修改到期日
- [ ] 缺少日期时不能直接入库
- [ ] 明确填写 `unknown` 后可以入库
- [ ] 确认后库存事务写入成功
- [ ] 识别草稿状态更新为 confirmed
- [ ] 模型版本写入库存
- [ ] 重复确认不会重复写入

### 库存管理闭环

- [ ] 查询全部有效库存
- [ ] 查询临期库存
- [ ] 查询过期库存
- [ ] 修改库存记录
- [ ] 消耗库存记录
- [ ] 删除库存记录
- [ ] 修改操作需要确认
- [ ] 消耗操作需要确认
- [ ] 删除操作需要确认
- [ ] 过期确认动作被拒绝
- [ ] 非所有者不能操作库存

### 菜谱闭环

- [ ] 有库存时能够生成菜谱
- [ ] 临期食材优先使用
- [ ] 库存食材和外购食材明确区分
- [ ] 空库存时返回合理提示
- [ ] 菜谱生成不会自动扣减库存

### Placeholder

- [ ] 发送不支持的需求
- [ ] 返回当前能力边界
- [ ] 原始请求写入 `unhandled_intents`
- [ ] 记录失败时仍能发送兜底回复

---

## P1：重启与故障恢复

- [ ] API 重启后数据库仍可访问
- [ ] Worker 重启后 queued 任务仍存在
- [ ] Worker 恢复超过 15 分钟的 running 任务
- [ ] 失败任务按指数退避重试
- [ ] 达到最大尝试次数后任务标记 failed
- [ ] 卡片确认中断后重启 Worker
- [ ] 重启后通过 checkpoint 恢复执行
- [ ] ChromaDB 重启后知识仍存在
- [ ] 库存数据库重启后数据仍存在
- [ ] 模型缓存重启后无需重新下载
- [ ] Vision 不可用时任务正确重试
- [ ] Chat API 不可用时任务正确失败或降级
- [ ] Bitable 不可用时副作用状态正确记录

---

## P1：安全与运维

### 密钥安全

- [ ] `.env` 不提交 Git
- [ ] API Key 不写入 README
- [ ] API Key 不写入日志
- [ ] 飞书 Secret 不写入日志
- [ ] 定期轮换 Chat API Key
- [ ] 定期轮换飞书 App Secret
- [ ] 限制生产机器的文件访问权限

### 数据安全

- [ ] 明确飞书图片保留期限
- [ ] 明确识别草稿保留期限
- [ ] 明确训练样本保留期限
- [ ] 明确知识库内容删除流程
- [ ] 明确用户数据备份位置
- [ ] 明确云端训练数据删除流程
- [ ] 明确第三方模型供应商的数据保留策略

### 备份

- [ ] 备份 `data/self_evolution.db`
- [ ] 备份 `data/chroma`
- [ ] 备份 `data/langgraph_checkpoints.db`
- [ ] 备份训练数据版本
- [ ] 备份模型 Adapter
- [ ] 备份最终部署模型
- [ ] 测试从备份恢复

### 监控和日志

- [ ] 配置 API 日志轮转
- [ ] 配置 Worker 日志轮转
- [ ] 配置 Vision 日志轮转
- [ ] 监控磁盘空间
- [ ] 监控任务失败数量
- [ ] 监控任务队列积压
- [ ] 监控 Chat API 错误率
- [ ] 监控 Vision 推理延迟
- [ ] 监控 GPU 显存和温度
- [ ] 配置服务异常自动重启

---

## P2：后续增强

- [ ] 支持多人用户和 owner 数据隔离
- [ ] 支持群聊 `@Bot`
- [ ] 支持 PDF 和 Word 附件
- [ ] 支持飞书加密事件正文
- [ ] 增加 Web 管理后台
- [ ] 增加主动临期提醒
- [ ] 增加部分数量消耗
- [ ] 增加知识删除和更新
- [ ] 增加混合检索
- [ ] 增加 Reranker
- [ ] 增加主动学习和低置信度样本队列
- [ ] 增加模型 A/B 测试
- [ ] 增加模型自动回滚
- [ ] 将 SQLite 队列替换为支持多 Worker 的队列系统
- [ ] 增加记账 Agent
- [ ] 增加健康 Agent

---

## 推荐执行顺序

- [ ] 1. 完成 Planner、专业 Agent、Embedding 和 Vision 模型选型
- [ ] 2. 获取 Chat API 地址、密钥和模型 ID
- [ ] 3. 创建飞书应用和 Bot
- [ ] 4. 创建 Bitable 和所需字段
- [ ] 5. 获取飞书全部凭据和授权用户 open_id
- [ ] 6. 创建并填写 `.env`
- [ ] 7. 确定纯 Python 或 Docker 部署方式
- [ ] 8. 配置公网 HTTPS 回调地址
- [ ] 9. 启动 API 和 Worker
- [ ] 10. 验收飞书消息和灵感写入
- [ ] 11. 验收知识入库和 RAG 查询
- [ ] 12. 部署并验收最终视觉模型
- [ ] 13. 验收食材确认和库存闭环
- [ ] 14. 完成故障恢复、安全和备份测试
- [ ] 15. 开始收集微调数据并进行云端训练
