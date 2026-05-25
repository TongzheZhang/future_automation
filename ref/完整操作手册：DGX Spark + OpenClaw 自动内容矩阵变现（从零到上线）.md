# 完整操作手册：DGX Spark + OpenClaw 自动内容矩阵变现（从零到上线）

> Archived: 该文档保留作历史参考，当前仓库实现已不再按其中的 WordPress 路线推进。请以仓库根目录的 `README.md` 为唯一实施基准。

## 推荐方案：AI 内容矩阵 + 联盟营销

**结论先行**：所有方案里，**自动内容矩阵 + 联盟营销** 是性价比最高的选择。理由：
- 一次搭建，长期自动运行，无需持续跟客户沟通
- DGX Spark 本地推理让内容生产成本接近零（无 API 费用）
- 不涉及交易风险，法律合规性最好
- 3–4 个月后形成 SEO 复利，收入滚雪球式增长[^1]

***

## 总体架构

```
DGX Spark（本地模型推理）
       ↓
OpenClaw（每天定时触发）
       ↓
抓取热点关键词 → 生成 SEO 文章 → 发布到 WordPress → 插入联盟链接
       ↓
Google/百度自然流量 → 读者点击联盟链接 → 产生佣金收入
```

整条链路搭好后，你每天只需检查一次报告，机器自动运行。[^2]

***

## 第一阶段：准备工作（约 2 小时）

### 步骤 1：在 DGX Spark 上安装 Ollama + 本地模型

DGX Spark 官方提供了 Ollama 的 Docker 笔记本，支持远程访问。[^3]

```bash
# 1. 安装 Ollama（DGX Spark 已预装 Docker）
docker pull ollama/ollama
docker run -d --gpus all -p 11434:11434 ollama/ollama

# 2. 下载模型（推荐 DeepSeek-R1 70B，DGX Spark 128GB 内存足够跑）
ollama pull deepseek-r1:70b

# 3. 验证运行正常
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:70b",
  "messages": [{"role": "user", "content": "写一段 SEO 测试文章"}],
  "stream": false
}'
```

模型下载完成后，Ollama 会自动提供 OpenAI 兼容的本地 API（`http://localhost:11434`）。[^4]

### 步骤 2：确认 OpenClaw 正常运行

假设你已按之前的方案安装了 OpenClaw，在 Terminal 里运行：

```bash
openclaw --version
openclaw start
```

打开 OpenClaw 的 Web 界面（默认 `http://localhost:3000`），在设置里将模型提供商切换为 **Ollama**，填入 `http://localhost:11434`，模型选 `deepseek-r1:70b`。[^5]

***

## 第二阶段：搭建 WordPress 站点（约 1 小时）

### 步骤 3：购买域名和主机

- **域名**：在阿里云万网购买（¥60–100/年），选你要做的细分领域，如 `techdaily.com` 或 `aitools.guide`
- **主机**：Hostinger VPS（约 $5.99/月）或阿里云轻量应用服务器（¥24/月），安装 WordPress 模板[^5]

### 步骤 4：安装 WordPress 必备插件

登录 WordPress 后台，安装以下插件：

| 插件 | 用途 |
|------|------|
| **Rank Math SEO** | 自动 SEO 优化，填充 meta 标签 |
| **Thirsty Affiliates** | 联盟链接管理和伪装（让链接更美观）[^6] |
| **WP REST API** | 允许 OpenClaw 通过 API 发布文章（WordPress 默认已启用）[^5] |

### 步骤 5：生成 WordPress Application Password

1. 进入 WordPress 后台 → 用户 → 个人资料
2. 滑到底部找 **Application Passwords** → 填写名称（如 `OpenClaw`）→ 点 **Add New Application Password**
3. 复制生成的密码（只显示一次）

***

## 第三阶段：连接 OpenClaw 到 WordPress（约 30 分钟）

### 步骤 6：安装 WordPress Skill

在 OpenClaw 的聊天框里输入：[^7][^5]

```
在 GitHub 上找到并安装 WordPress agent skill
```

OpenClaw 会自动从 GitHub 下载并安装。安装完成后，它会询问：
1. **WordPress 网站 URL**：填你的域名（如 `https://techdaily.com`）
2. **WordPress 用户名**：你的管理员账户名
3. **Application Password**：步骤 5 里复制的密码

填完后 OpenClaw 会自动测试连接。成功后你可以直接在聊天框测试：

```
写一篇关于"2026年最好用的 AI 写作工具"的 SEO 文章，发布到 WordPress，分类设为 "AI Tools"
```

看到 WordPress 后台出现文章就说明打通了。[^5]

***

## 第四阶段：配置自动化心跳（核心步骤，约 1 小时）

### 步骤 7：创建选题来源

在 DGX Spark 上创建一个文件夹 `~/content-pipeline/`，里面建几个文件：

**`~/content-pipeline/topics.md`**（每天补充关键词）：
```markdown
# 待写主题
- 2026年最值得买的AI编程工具对比
- DeepSeek vs GPT-4o：哪个更适合中文写作
- 家用 NAS 存储方案推荐（含亚马逊链接）
- 轻量跑鞋推荐 2026（京东联盟）
```

**`~/content-pipeline/affiliate-config.md`**（你的联盟账号信息）：
```markdown
# 联盟配置
- 亚马逊联盟 ID：amzn-xxxx
- 京东联盟 ID：jd-xxxx
- 目标每篇文章插入 2-3 个联盟链接
- 链接格式使用 Thirsty Affiliates 短链
```

### 步骤 8：配置 HEARTBEAT.md

在 OpenClaw 聊天框里输入：

```
帮我创建一个 HEARTBEAT.md，每天早上 8:00 自动执行以下任务：
1. 读取 ~/content-pipeline/topics.md 里的第一条待写主题
2. 搜索这个主题的竞品文章（排名前5）
3. 生成一篇 1500字的 SEO 文章，包含：H2/H3 标题结构、自然插入关键词、2-3个联盟商品推荐
4. 将文章发布到 WordPress（草稿状态，等我审核后手动发布）
5. 把已完成的主题从 topics.md 移到 published.md
6. 通过 Telegram 发送完成通知给我
```

OpenClaw 会自动生成 HEARTBEAT.md 文件。生成后输入：

```
enable heartbeat.md
```

自动化正式启动。[^8]

### 步骤 9：（可选）配置 Cron 精确定时任务

HEARTBEAT 每 30 分钟检查一次，如果你需要精确在每天 8:00 触发，用 Cron：[^9]

```bash
openclaw cron add \
  --name "daily-content" \
  --cron "0 8 * * 1-5" \
  --tz "Asia/Shanghai" \
  --session isolated \
  --message "从 ~/content-pipeline/topics.md 读取今日主题，生成并发布 SEO 文章到 WordPress" \
  --model deepseek-r1:70b \
  --announce \
  --channel telegram
```

***

## 第五阶段：加入联盟计划（约 1 小时）

### 步骤 10：注册联盟账号

根据你的内容方向注册对应的联盟计划：

| 平台 | 申请网址 | 佣金范围 | 适合方向 |
|------|---------|---------|---------|
| **亚马逊联盟**（海外） | associates.amazon.com | 3–10% | 科技产品、书籍、家居 |
| **京东联盟** | union.jd.com | 2–50% | 3C、家电、运动 |
| **阿里妈妈**（淘宝） | alimama.com | 3–70% | 服装、日用品 |
| **ShareASale**（软件类） | shareasale.com | 20–50% | SaaS 工具、在线课程 |

**推荐先选一个细分领域**，比如「AI 工具评测」，只接入 ShareASale（很多 SaaS 产品佣金高达 30%）。

### 步骤 11：在 Thirsty Affiliates 中添加链接

1. WordPress 后台 → Thirsty Affiliates → Add New Link
2. 填入联盟商品原始链接
3. 设置短链（如 `yourdomain.com/go/chatgpt`）
4. 在 `affiliate-config.md` 里记录这些短链，OpenClaw 写文章时会自动引用

***

## 第六阶段：监控与扩量

### 每天只需做的事（10 分钟）

1. 打开 Telegram，查看 OpenClaw 发来的当日文章通知
2. 快速浏览文章质量，满意则在 WordPress 后台点「发布」
3. 向 `topics.md` 补充 1–2 个新主题

### 第 1 个月目标

- 发布 30–40 篇文章（每天 1–2 篇）
- 提交 Google Search Console（让 Google 收录）
- 检查哪类文章点击率最高，让 OpenClaw 多写那类主题

### 第 2–3 个月扩量

当有文章进入 Google 前 10 后，开始扩量：[^10]

```bash
# 增加每日发文量到 3 篇
openclaw cron add --name "evening-content" --cron "0 20 * * *" ...

# 增加社交媒体分发：让 OpenClaw 把文章摘要自动发到小红书/Twitter
```

***

## 预期收入时间线

| 时间节点 | 内容量 | 预期月收入 | 关键里程碑 |
|---------|-------|----------|----------|
| 第 1 个月 | 30 篇 | $0–50 | 建站、SEO 权重积累期 |
| 第 2 个月 | 60 篇 | $50–200 | 首批文章开始被收录 |
| 第 3 个月 | 90 篇 | $200–800 | 部分关键词进入首页[^11] |
| 第 4–6 个月 | 150–200 篇 | $500–3,000+ | SEO 复利效应显现 |

***

## 常见问题与解决方案

**Q：OpenClaw 写出来的文章质量不够好怎么办？**

在 HEARTBEAT.md 的 prompt 里加入具体要求，例如：「文章需包含真实用户使用场景、具体数据对比、亲测评价口吻，避免泛泛而谈」。使用 DeepSeek-R1 70B 开启 `--thinking high` 参数可显著提升文章深度。[^9]

**Q：担心 Google 惩罚 AI 生成内容？**

Google 目前的政策是针对「无价值内容」，而非所有 AI 内容。关键是提供独特价值：加入真实产品对比数据、用户评价摘要、价格趋势图表。让 OpenClaw 的 prompt 包含「搜集最新用户评价和价格数据」指令。[^12]

**Q：没有海外银行卡，收不到亚马逊佣金怎么办？**

优先用**京东联盟**（微信/支付宝提现）或**阿里妈妈**（支付宝直接提现），完全不需要海外账户。

---

## References

1. [OpenClaw Passive Income Strategy That Actually Works - YouTube](https://www.youtube.com/watch?v=5asShWKXVLc) - Learn how to make money with OpenClaw and AI automation in this 2026 monetization guide! Discover ho...

2. [OpenClaw Marketing Automation: Run AI Campaigns 24/7 | Blink Blog](https://blink.new/blog/openclaw-ai-marketing-campaigns-guide-2026) - OpenClaw marketing automation lets you run AI campaigns around the clock — social posts, newsletters...

3. [Run Local AI Models with Ollama + NVIDIA DGX Spark - YouTube](https://www.youtube.com/watch?v=yOgNv4HrYZ4) - For more information, or to buy a NVIDIA DGX Spark: https://nvda.ws/4nFFtPT Code: ...

4. [How to Set Up and Run DeepSeek-R1 Locally With Ollama](https://www.datacamp.com/tutorial/deepseek-r1-ollama) - Setting Up DeepSeek-R1 Locally With Ollama · Step 1: Install Ollama · Step 2: Download and run DeepS...

5. [How to set up OpenClaw for WordPress - Hostinger](https://www.hostinger.com/tutorials/how-to-set-up-openclaw-for-wordpress) - Setting up OpenClaw for WordPress involves three main stages: deploying OpenClaw on a server, instal...

6. [Affiliate Marketing Tutorial for Beginners 2026 (Step By Step with AI)](https://www.youtube.com/watch?v=SJwwe1YXisA) - Thank you for being real in your videos and not AI even when you teach AI. Ive been making $$$$ sinc...

7. [OpenClaw + WordPress Guide : I tried it & the potential is INFINITE](https://www.youtube.com/watch?v=-YGidOU43iE) - Use OpenClaw for WordPress to automate your marketing tasks with AI https://hostingeracademy.com/4s2...

8. [Stop Wasting Time & Master Openclaw in 12 Min - YouTube](https://www.youtube.com/watch?v=iSLruYDGT58) - Use OpenClaw Safely Using Hostinger https://youricreates.com/clawdbot In this video, I break down ho...

9. [Cron vs Heartbeat - OpenClaw Docs](https://docs.openclaw.ai/automation/cron-vs-heartbeat) - Multiple periodic checks: Instead of 5 separate cron jobs checking inbox, calendar, weather, notific...

10. [2026 AI Citation Position & Revenue Report - The Digital Bloom](https://thedigitalbloom.com/learn/ai-citation-position-revenue-report-2026/) - This report maps the complete chain from SERP position to AI citation probability to conversion prem...

11. [OpenClaw内容营销自动化矩阵完全解密 - AtomGit开源社区](https://gitcode.csdn.net/69c1b7eb54b52172bc63cb66.html) - 每月仅需$30成本即可创造$3200收入的OpenClaw内容营销自动化系统，通过AI写手DeepSeek-V3实现全流程自动化运营。该系统每天自动抓取热点话题， ...

12. [The Complete Guide to AI SEO Agents (2026) | MEGA SEO](https://www.gomega.ai/blog/complete-guide-ai-seo-agents/) - This guide covers everything you need to know: what AI SEO agents are, how they work under the hood,...
