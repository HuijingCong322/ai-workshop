# AI Newspaper Agent 合并版说明

这个文件把项目里真正有用的代码主线合并到一个地方，并用中文解释每一部分的作用。它不是替代所有 notebook 的可运行源码，而是一个便于答辩、复习和讲解的整合版。

项目主线可以理解为：

```text
用户
  ↓
News Generator Agent
  ↓ 调用 newspaper MCP tools
生成新闻、验证质量、发送邮件
  ↓ 调用 preference_agent.chat(...)
Preference Agent
  ↓ 调用 preference memory tools
检索/更新用户偏好，审核内容是否符合偏好
```

## 1. 两个 Agent 分工

### News Generator Agent

来源：`0401_news_agent_client.ipynb`

它是用户真正交互的 Agent，负责统筹整条流程：

```text
发现新闻
→ 创建 newspaper draft
→ 调用 Preference Agent 审核
→ 如果被拒绝，根据建议修改
→ 通过后 validate
→ validate 通过后 publish_newspaper 发邮件
→ 把结果归档
```

它使用两个 MCP server：

```python
servers=["newspaper", "preference_agent"]
```

也就是说它能调用 newspaper 工具，也能调用 Preference Agent 的 `chat()` 工具。

### Preference Agent

来源：`0301_multi_agent_collaboration.ipynb`

它不是主要给用户直接聊天的 Agent，而是内部审核者。它负责：

```text
读取用户偏好
检索 ChromaDB 记忆
审核 draft 是否符合用户偏好
返回 APPROVED 或 DENIED
给出具体修改建议
在观察到新偏好时写回 memory
```

对外暴露的核心工具是：

```python
@preference_agent_mcp.tool()
async def chat(message: str) -> str:
    return await agent(message)
```

注意：`message` 必须包含完整 draft 内容，因为两个 Agent 不共享上下文。

## 2. ChromaDB 和 RAG

项目里 ChromaDB 不是 `.md` 文件，而是本地向量数据库目录。

新闻文章记忆初始化：

```python
article_memory = ArticleMemoryService()
article_memory.initialize(settings.data_dir / "chromadb")
```

用户偏好记忆初始化：

```python
memory_service = MemoryService(collection_name="workshop_preferences")
memory_service.initialize(Path.cwd().parent / "data" / "learning-agent" / "chroma")
```

这个项目实现的是 Agentic RAG：

```text
文章/偏好写入 ChromaDB
→ 当前任务开始时做语义检索
→ 把检索结果放入 prompt
→ LLM 生成新闻摘要、审核建议或最终 newspaper
```

核心检索代码：

```python
related = article_memory.search_articles(
    query=article["content"],
    limit=5,
)
```

然后把检索结果放入 prompt：

```python
RELATED PAST COVERAGE:
{chr(10).join(f"- {r['title']}" for r in related[:3])}
```

这就是 RAG 的 `retrieval + augmentation + generation`。

## 3. Newspaper MCP Server 的关键代码

来源：`0202_advanced_mcp_smart_tools.ipynb`

### 初始化服务

```python
@asynccontextmanager
async def app_lifespan(mcp: FastMCP):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    hn_client = HackerNewsClient()
    interests_service = InterestsFileService(settings.data_dir)

    article_memory = ArticleMemoryService()
    article_memory.initialize(settings.data_dir / "chromadb")

    newspaper_service = NewspaperService(settings.data_dir)

    email_service = EmailService(
        {
            "server": "smtp.gmail.com",
            "port": 465,
            "use_ssl": True,
            "username": os.getenv("MCP_SMTP_FROM_EMAIL", ""),
            "password": os.getenv("MCP_SMTP_PASSWORD", ""),
            "from_email": os.getenv("MCP_SMTP_FROM_EMAIL", ""),
            "from_name": "AI Newspaper Agent",
        }
    )

    yield AppContext(
        hn_client=hn_client,
        interests_service=interests_service,
        article_memory=article_memory,
        newspaper_service=newspaper_service,
        email_service=email_service,
        settings=settings,
    )
```

### 发现新闻并存入 ChromaDB

```python
@mcp.tool()
async def discover_stories(
    query: str = "technology",
    count: int = 10,
    sources: list[str] = ["hn"],
    ctx: Context = None,
) -> str:
    hn_client = ctx.request_context.lifespan_context.hn_client
    article_memory = ctx.request_context.lifespan_context.article_memory
    interests_service = ctx.request_context.lifespan_context.interests_service

    interests = interests_service.read_interests()
    user_topics = interests.get("topics", [])

    story_ids = await hn_client.get_story_ids("topstories", count)
    enriched_stories = []

    for story_id in story_ids:
        story = await hn_client.get_item(story_id)
        if not story or not story.get("title") or not story.get("url"):
            continue

        full_content = await fetch_content(story["url"])
        content_id = f"cnt_hn_{story_id}"

        topics = [
            topic for topic in user_topics
            if topic.lower() in story["title"].lower()
        ]

        article_memory.store_article_with_content_id(
            content_id=content_id,
            url=story["url"],
            content=full_content,
            title=story["title"],
            source="hn",
            topics=topics,
            summary="",
        )

        related = article_memory.search_articles(
            query=story["title"],
            limit=3,
        )

        enriched_stories.append(
            {
                "content_id": content_id,
                "title": story["title"],
                "url": story["url"],
                "related_past_articles": [a["title"] for a in related],
                "topics": topics,
            }
        )

    return str(enriched_stories)
```

### 用 RAG 生成摘要并加入 newspaper

```python
@mcp.tool()
async def add_content_cluster(
    newspaper_id: str,
    section: str,
    content_ids: list[str],
    treatment: str = "brief",
    ctx: Context = None,
) -> str:
    article_memory = ctx.request_context.lifespan_context.article_memory
    newspaper_service = ctx.request_context.lifespan_context.newspaper_service
    interests_service = ctx.request_context.lifespan_context.interests_service

    interests = interests_service.read_interests()

    for content_id in content_ids:
        article = article_memory.get_by_content_id(content_id)
        if not article:
            continue

        related = article_memory.search_articles(
            query=article["content"],
            limit=5,
        )

        sample_result = await ctx.sample(
            messages=[
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"""Summarize this article in {treatment} style.

USER INTERESTS:
{chr(10).join(f"- {topic}" for topic in interests.get("topics", []))}

ARTICLE:
Title: {article["title"]}
Content: {article["content"][:3000]}

RELATED PAST COVERAGE:
{chr(10).join(f"- {r['title']}" for r in related[:3])}
""",
                    },
                }
            ]
        )

        newspaper_service.add_article(
            newspaper_id=newspaper_id,
            section_title=section,
            article_data={
                "title": article["title"],
                "content": sample_result.text,
                "url": article["url"],
                "related": [r["title"] for r in related[:3]],
            },
        )

    return "Content cluster added"
```

### 验证，不通过就返回修改建议

```python
@mcp.tool()
async def validate_and_finalize(
    newspaper_id: str,
    min_reading_time: int = None,
    min_articles: int = None,
    ctx: Context = None,
) -> str:
    newspaper_service = ctx.request_context.lifespan_context.newspaper_service
    newspaper_data = newspaper_service.get_newspaper_data(newspaper_id)

    if not newspaper_data:
        return "Newspaper not found"

    issues = []

    current_reading_time = newspaper_data["metadata"]["total_reading_time"]
    current_articles = newspaper_data["metadata"]["article_count"]

    if min_reading_time and current_reading_time < min_reading_time:
        issues.append(
            f"Reading time too short. Add more detailed articles."
        )

    if min_articles and current_articles < min_articles:
        issues.append(
            f"Too few articles. Add more content IDs."
        )

    if not issues:
        return "APPROVED BY VALIDATION: ready to publish"

    return "VALIDATION FAILED:\n" + "\n".join(issues)
```

### 发布并归档

```python
@mcp.tool()
async def publish_newspaper(
    newspaper_id: str,
    delivery_method: str = "email",
    ctx: Context = None,
) -> str:
    newspaper_service = ctx.request_context.lifespan_context.newspaper_service
    email_service = ctx.request_context.lifespan_context.email_service
    article_memory = ctx.request_context.lifespan_context.article_memory

    newspaper_data = newspaper_service.get_newspaper_data(newspaper_id)
    if not newspaper_data:
        return "Newspaper not found"

    email_sent = False
    if delivery_method in ["email", "both"]:
        result = email_service.send_newspaper(newspaper_data, version=2)
        email_sent = result["success"]

    article_memory.store_newspaper(newspaper_id, newspaper_data)

    return f"Published. Email sent: {email_sent}. Stored in memory."
```

## 4. Preference Memory Tools 的关键代码

来源：`0301_multi_agent_collaboration.ipynb`

```python
@preference_tools_server.tool()
async def store_preference(
    content: str,
    metadata: dict | None = None,
) -> dict:
    return memory_service.store_document(
        content=content,
        metadata=metadata or {"type": "preference"},
    )
```

```python
@preference_tools_server.tool()
async def search_preferences(
    query: str,
    limit: int = 5,
    metadata: dict | None = None,
) -> list:
    return memory_service.search_documents(
        query,
        limit,
        metadata_filter=metadata,
    )
```

```python
@preference_tools_server.tool()
async def add_interests(topics: list[str]) -> str:
    for topic in topics:
        existing = memory_service.search_documents(
            query=topic,
            limit=1,
            metadata_filter={"type": "interest"},
        )

        if not existing or existing[0].get("distance", 1) > 0.5:
            memory_service.store_document(
                content=topic,
                metadata={"type": "interest"},
            )

    return "Interests updated"
```

## 5. Preference Agent 的关键代码

```python
@preference_agent_app.agent(
    instruction="""
You are a Preference Modeling Specialist.

Responsibilities:
1. Review content drafts against stored user preferences.
2. Search preferences before judging.
3. End every review with APPROVED or DENIED.
4. If denied, provide concrete fixes.
5. If new feedback appears, call store_preference.
""",
    name="Preference Analyst",
    servers=["preferences"],
)
async def preference_analyst():
    pass
```

包装成 MCP tool：

```python
@preference_agent_mcp.tool()
async def chat(message: str) -> str:
    return await agent(message)
```

## 6. News Generator Agent 的关键代码

来源：`0401_news_agent_client.ipynb`

```python
NEWS_AGENT_INSTRUCTION = """
You are the user-facing News Generator Agent.

Workflow:
1. Use newspaper tools to discover stories and create a draft.
2. Before publishing, call preference_agent.chat with the complete draft.
3. If DENIED, revise using feedback and review again.
4. Try at most 3 review rounds.
5. If APPROVED, call validate_and_finalize.
6. If valid, call publish_newspaper.
7. After publishing, send a delivery summary to preference_agent.chat so it can store useful preference patterns.

Do not ask the user to call Preference Agent directly.
Do not publish denied or invalid drafts.
"""
```

```python
@news_agent_app.agent(
    name="News Generator",
    instruction=NEWS_AGENT_INSTRUCTION,
    servers=["newspaper", "preference_agent"],
    request_params=RequestParams(max_iterations=80),
)
async def news_generator():
    pass


async def ask_news_agent(user_request: str) -> str:
    async with news_agent_app.run() as agent:
        return await agent(user_request)
```

用户只需要调用：

```python
response = await ask_news_agent(
    "Create a short morning AI news brief for me. Review it against my preferences, revise if needed, and send it by email only after approval."
)

print(response)
```

## 7. 运行顺序

```text
1. 配置 OPENROUTER_API_KEY、MCP_SMTP_FROM_EMAIL、MCP_SMTP_PASSWORD
2. 运行 0202_advanced_mcp_smart_tools.ipynb，启动 newspaper server: http://localhost:8080/mcp
3. 运行 0301_multi_agent_collaboration.ipynb，启动 preference agent server: http://localhost:8082/mcp
4. 运行 0401_news_agent_client.ipynb，修改最后一个 ask_news_agent(...) 请求
```

## 8. 当前项目边界

这个项目已经有完整的 Agent 原型链路：

```text
News Generator Agent
+ Preference Agent
+ MCP tools
+ ChromaDB memory
+ RAG-style retrieval
+ email publishing
```

但它还不是完整产品化系统：

- 主要以 notebook 运行。
- 没有网页 UI。
- 自动返工主要依赖 Agent 指令，不是硬编码 workflow controller。
- 上下文溢出主要靠 top-k 检索和截断预防，没有完整异常恢复机制。
