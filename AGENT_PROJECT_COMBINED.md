# AI Newspaper Agent 项目合并版代码说明

这个文件把项目里真正有用的主线代码合并到一起，并用中文解释每一部分在做什么。

你的项目可以理解成两个 Agent 加两个工具层：

```text
用户
  ↓
News Generator Agent
  ↓ 使用 newspaper MCP tools
新闻发现、生成、验证、发送邮件
  ↓ 调用 preference_agent.chat(...)
Preference Agent
  ↓ 使用 preference memory tools
读取/更新用户偏好，审核内容是否符合偏好
```

也就是说：

- `News Generator Agent`：面向用户，负责生成和发布新闻。
- `Preference Agent`：内部审核者，负责查偏好、给 `APPROVED / DENIED`、更新记忆。
- `ChromaDB`：用来存文章记忆和用户偏好记忆。
- `MCP tools`：Agent 之间和 Agent 与工具之间通信的接口。

---

## 1. Newspaper MCP Server：新闻生成工具层

来源文件：`0202_advanced_mcp_smart_tools.ipynb`

这一层不是用户直接聊天的 Agent，而是给 News Generator Agent 使用的工具集合。

它负责：

- 抓取 Hacker News 文章。
- 把文章内容存入 ChromaDB。
- 从 ChromaDB 检索相关文章。
- 使用 LLM 生成摘要和 newspaper draft。
- 验证内容质量。
- 通过邮件发送，并归档结果。

### 1.1 初始化服务和 ChromaDB

```python
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastmcp import FastMCP, Context

from src.server.config.settings import get_settings
from src.server.services.article_memory_v2 import ArticleMemoryService
from src.server.services.email_service import EmailService
from src.server.services.http_client import HackerNewsClient, fetch_content
from src.server.services.interests_file import InterestsFileService
from src.server.services.newspaper_service import NewspaperService


@dataclass
class AppContext:
    """所有工具共享的上下文。MCP 工具通过 ctx 访问这些服务。"""

    hn_client: HackerNewsClient
    interests_service: InterestsFileService
    article_memory: ArticleMemoryService
    newspaper_service: NewspaperService
    email_service: EmailService
    settings: object


@asynccontextmanager
async def app_lifespan(mcp: FastMCP):
    """启动 MCP server 时初始化所有服务。"""
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    hn_client = HackerNewsClient()
    interests_service = InterestsFileService(settings.data_dir)

    # 这里初始化 ChromaDB。它不是 .md 文件，而是一个本地向量数据库目录。
    article_memory = ArticleMemoryService()
    article_memory.initialize(settings.data_dir / "chromadb")

    newspaper_service = NewspaperService(settings.data_dir)

    email_service = EmailService(
        {
            "server": "smtp.gmail.com",
            "port": 465,
            "use_tls": False,
            "use_ssl": True,
            "username": os.getenv("MCP_SMTP_FROM_EMAIL", ""),
            "password": os.getenv("MCP_SMTP_PASSWORD", ""),
            "from_email": os.getenv("MCP_SMTP_FROM_EMAIL", ""),
            "from_name": "AI Newspaper Agent",
        }
    )

    try:
        yield AppContext(
            hn_client=hn_client,
            interests_service=interests_service,
            article_memory=article_memory,
            newspaper_service=newspaper_service,
            email_service=email_service,
            settings=settings,
        )
    finally:
        print("MCP server stopped")


mcp = FastMCP(
    name="advanced-newspaper-agent",
    instructions="""
    Advanced newspaper creation agent.
    It can discover stories, store article memory, create newspapers,
    validate quality, publish by email, and archive results.
    """,
    lifespan=app_lifespan,
)
```

### 1.2 发现新闻，并存入 ChromaDB

这一步是 RAG 的“存储层”。系统先把文章全文保存到 ChromaDB，后面生成时再按需检索。

```python
@mcp.tool()
async def discover_stories(
    query: str = "technology",
    count: int = 10,
    sources: list[str] = ["hn"],
    ctx: Context = None,
) -> str:
    """
    发现候选新闻。

    关键点：
    1. 从 Hacker News 抓取文章。
    2. 获取文章正文。
    3. 生成 content_id。
    4. 把文章存入 ChromaDB。
    5. 返回可供 Agent 选择的候选文章列表。
    """
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

        title_lower = story["title"].lower()
        topics = [topic for topic in user_topics if topic.lower() in title_lower]

        # 存入 ChromaDB：content + metadata。
        article_memory.store_article_with_content_id(
            content_id=content_id,
            url=story["url"],
            content=full_content,
            title=story["title"],
            source="hn",
            topics=topics,
            summary="",
        )

        # 立刻检索相似历史内容，作为后续生成时的参考。
        related = article_memory.search_articles(query=story["title"], limit=3)

        enriched_stories.append(
            {
                "content_id": content_id,
                "title": story["title"],
                "url": story["url"],
                "source": "hn",
                "score": story.get("score", 0),
                "related_past_articles": [a["title"] for a in related],
                "topics": topics,
            }
        )

    result = "# Discovered Stories\n\n"
    for item in enriched_stories:
        result += f"- {item['content_id']}: {item['title']}\n"
        result += f"  URL: {item['url']}\n"
        if item["related_past_articles"]:
            result += f"  Related: {', '.join(item['related_past_articles'])}\n"

    return result
```

### 1.3 用检索结果增强生成：RAG 核心

这里是真正体现 RAG 的地方：

```text
content_id
  ↓
从 ChromaDB 取文章正文
  ↓
用文章正文搜索相关历史文章
  ↓
把用户偏好 + 当前文章 + 相关历史报道放进 prompt
  ↓
LLM 生成摘要
```

```python
@mcp.tool()
async def add_content_cluster(
    newspaper_id: str,
    section: str,
    content_ids: list[str],
    treatment: str = "brief",
    auto_enhance: bool = True,
    link_related: bool = True,
    ctx: Context = None,
) -> str:
    """把一组文章加入 newspaper，并用 RAG 生成个性化摘要。"""
    article_memory = ctx.request_context.lifespan_context.article_memory
    newspaper_service = ctx.request_context.lifespan_context.newspaper_service
    interests_service = ctx.request_context.lifespan_context.interests_service

    interests = interests_service.read_interests()

    articles = []
    for content_id in content_ids:
        article = article_memory.get_by_content_id(content_id)
        if article:
            articles.append(article)

    if not articles:
        return "No valid articles found"

    added_articles = []

    for article in articles:
        # RAG retrieval：根据当前文章内容检索历史相关文章。
        related = article_memory.search_articles(query=article["content"], limit=5)

        # RAG augmentation：把检索结果放入 LLM prompt。
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
{chr(10).join(f"- {r["title"]}" for r in related[:3])}

Write a clear, useful summary for a personalized newspaper.
""",
                    },
                }
            ]
        )

        summary = sample_result.text

        newspaper_service.add_article(
            newspaper_id=newspaper_id,
            section_title=section,
            article_data={
                "title": article["title"],
                "content": summary,
                "url": article["url"],
                "source": article.get("source", ""),
                "related": [r["title"] for r in related[:3]],
            },
        )

        added_articles.append(article["title"])

    return f"Added {len(added_articles)} articles to {section}: {', '.join(added_articles)}"
```

### 1.4 验证：不通过就返回修改建议

```python
@mcp.tool()
async def validate_and_finalize(
    newspaper_id: str,
    min_reading_time: int = None,
    min_articles: int = None,
    ctx: Context = None,
) -> str:
    """
    发布前质量检查。

    如果不通过，不会直接发送邮件，而是返回具体问题和修复建议。
    News Generator Agent 会根据这些建议修改 draft。
    """
    newspaper_service = ctx.request_context.lifespan_context.newspaper_service
    newspaper_data = newspaper_service.get_newspaper_data(newspaper_id)

    if not newspaper_data:
        return "Newspaper not found"

    result = newspaper_service.validate(newspaper_id)
    issues = []

    current_reading_time = newspaper_data["metadata"]["total_reading_time"]
    current_articles = newspaper_data["metadata"]["article_count"]

    if min_reading_time and current_reading_time < min_reading_time:
        shortfall = min_reading_time - current_reading_time
        issues.append(
            {
                "type": "reading_time",
                "current": current_reading_time,
                "required": min_reading_time,
                "suggestion": f"Add {shortfall // 3} more detailed articles",
                "fix": f"Use add_content_cluster() with more content IDs and treatment='detailed'",
            }
        )

    if min_articles and current_articles < min_articles:
        shortfall = min_articles - current_articles
        issues.append(
            {
                "type": "article_count",
                "current": current_articles,
                "required": min_articles,
                "suggestion": f"Add {shortfall} more articles",
                "fix": f"Use add_content_cluster() with {shortfall} more content IDs",
            }
        )

    all_issues = result.get("issues", []) + [i["type"] for i in issues]

    if not all_issues:
        return "# APPROVED BY VALIDATION\n\nNewspaper is valid and ready to publish."

    output = "# VALIDATION FAILED\n\n"
    for issue in issues:
        output += f"- Problem: {issue['type']}\n"
        output += f"  Current: {issue['current']}\n"
        output += f"  Required: {issue['required']}\n"
        output += f"  Suggestion: {issue['suggestion']}\n"
        output += f"  Fix: {issue['fix']}\n"

    output += "\nDo not publish until these issues are fixed."
    return output
```

### 1.5 发布：发邮件并归档

```python
@mcp.tool()
async def publish_newspaper(
    newspaper_id: str,
    delivery_method: str = "email",
    ctx: Context = None,
) -> str:
    """
    发布 newspaper。

    做三件事：
    1. 生成 HTML。
    2. 发送邮件。
    3. 把最终 newspaper 存入 ChromaDB 记忆系统。
    """
    newspaper_service = ctx.request_context.lifespan_context.newspaper_service
    email_service = ctx.request_context.lifespan_context.email_service
    article_memory = ctx.request_context.lifespan_context.article_memory
    settings = ctx.request_context.lifespan_context.settings

    newspaper_data = newspaper_service.get_newspaper_data(newspaper_id)
    if not newspaper_data:
        return "Newspaper not found"

    html_content = email_service._create_html_version(newspaper_data)
    html_file = settings.data_dir / "newspapers" / f"{newspaper_id}.html"
    html_file.parent.mkdir(parents=True, exist_ok=True)
    html_file.write_text(html_content, encoding="utf-8")

    email_sent = False
    if delivery_method in ["email", "both"]:
        result = email_service.send_newspaper(newspaper_data, version=2)
        email_sent = result["success"]

    # 存储最终推荐结果/报纸结果。
    article_memory.store_newspaper(newspaper_id, newspaper_data)

    output = "# Newspaper Published\n\n"
    output += f"Title: {newspaper_data['title']}\n"
    output += f"Articles: {newspaper_data['metadata']['article_count']}\n"
    output += f"Reading time: {newspaper_data['metadata']['total_reading_time']} minutes\n"
    output += f"HTML: {html_file}\n"
    output += f"Email sent: {email_sent}\n"
    output += "Archive: stored in memory\n"

    return output
```

---

## 2. Preference Memory Tools：用户偏好记忆工具层

来源文件：`0301_multi_agent_collaboration.ipynb`

这一层是 Preference Agent 使用的工具。它直接和 ChromaDB 交互，负责存储和检索用户偏好。

### 2.1 初始化偏好记忆库

```python
from services.memory_service import MemoryService

# collection_name 是 ChromaDB 里的集合名。
memory_service = MemoryService(collection_name="workshop_preferences")

# 这里指定 ChromaDB 本地存储路径。
memory_service.initialize(Path.cwd().parent / "data" / "learning-agent" / "chroma")
```

### 2.2 初始化一些用户偏好

```python
pref_doc = """User preferences:
- Primary interests: Agentic AI, Machine Learning, Python development
- Content depth: Prefers technical deep-dives
- Time patterns: In morning
- Sentiment: Analytical, educational tone
"""

memory_service.store_document(
    content=pref_doc,
    doc_id="user_preferences_v1",
    metadata={"type": "preferences", "version": 1},
)
```

### 2.3 暴露成 MCP tools

```python
preference_tools_server = FastMCP(
    name="preference-tools-server",
    instructions="""
    Tools for storing and searching user preferences.
    These tools are backed by ChromaDB.
    """,
)


@preference_tools_server.tool()
async def store_preference(
    content: str,
    metadata: dict | None = None,
) -> dict:
    """把新的用户偏好写入 ChromaDB。"""
    return memory_service.store_document(
        content=content,
        metadata=metadata or {"type": "preference"},
    )


@preference_tools_server.tool()
async def search_preferences(
    query: str,
    limit: int = 5,
    metadata: dict | None = None,
) -> list:
    """用语义检索查找相关用户偏好。"""
    return memory_service.search_documents(
        query,
        limit,
        metadata_filter=metadata,
    )


@preference_tools_server.tool()
async def read_interests() -> str:
    """读取用户兴趣。"""
    results = memory_service.search_documents(
        query="user interests topics preferences",
        limit=10,
        metadata_filter={"type": "interest"},
    )

    if not results:
        return "No interests stored yet."

    topics = [doc.get("content", "").strip() for doc in results if doc.get("content")]

    output = "# User Interests\n\n"
    for topic in topics:
        output += f"- {topic}\n"

    return output


@preference_tools_server.tool()
async def add_interests(topics: list[str]) -> str:
    """
    添加用户兴趣。

    这里不是简单字符串去重，而是先做语义搜索，避免存入重复或高度相似的兴趣。
    """
    added = []

    for topic in topics:
        existing = memory_service.search_documents(
            query=topic,
            limit=1,
            metadata_filter={"type": "interest"},
        )

        if not existing or existing[0].get("distance", 1) > 0.5:
            doc_id = f"interest_{topic.lower().replace(' ', '_')}"
            memory_service.store_document(
                content=topic,
                doc_id=doc_id,
                metadata={"type": "interest"},
            )
            added.append(topic)

    return f"Added interests: {', '.join(added)}" if added else "No new interests added"
```

---

## 3. Preference Agent：偏好审核 Agent

来源文件：`0301_multi_agent_collaboration.ipynb`

Preference Agent 不负责生成新闻，它负责审核：

```text
这份新闻是否符合用户偏好？
如果不符合，具体要怎么改？
如果符合，返回 APPROVED。
如果观察到新的用户偏好，把它写入记忆系统。
```

```python
from fast_agent import FastAgent, RequestParams


preference_agent_app = FastAgent(
    "Preference Agent",
    config_path=str(setup_fastagent_config(preference_tools_server_url)),
)


@preference_agent_app.agent(
    instruction=f"""You are a PREFERENCE MODELING SPECIALIST.

YOUR RESPONSIBILITIES:
1. Review content drafts against stored user preferences.
2. Always end reviews with explicit "APPROVED" or "DENIED: [specific reasons]".
3. Use search_preferences and read_interests before judging.
4. When you observe useful feedback, call store_preference.

WHEN DENYING:
- Give 2-3 concrete fixes.
- Explain which user preference was violated.

WHEN APPROVING:
- Mention what aligned well.
- Keep the verdict clear.
""",
    name="Preference Analyst",
    servers=["preferences"],
    request_params=RequestParams(max_iterations=30),
)
async def preference_analyst():
    pass
```

### 3.1 把 Preference Agent 包装成一个 MCP 工具

News Generator Agent 不能直接共享 Preference Agent 的上下文，所以通过 `chat(message)` 把完整 draft 发过去。

```python
preference_agent_mcp = FastMCP(
    name="preference-agent",
    instructions="""
    Preference Agent.
    Use chat(message) to review complete drafts or store feedback.
    """,
)


@preference_agent_mcp.tool()
async def chat(message: str) -> str:
    """
    给 Preference Agent 发消息。

    注意：message 必须包含完整 draft。
    不能只写“帮我审核刚才那个草稿”，因为两个 Agent 不共享上下文。
    """
    return await agent(message)
```

---

## 4. News Generator Agent：用户真正交互的 Agent

来源文件：`0401_news_agent_client.ipynb`

这是你后来补的用户入口。用户不需要手动调用 Preference Agent。

用户只和 News Generator Agent 交互：

```python
response = await ask_news_agent(
    "Create a short morning AI news brief for me. Review it against my preferences, revise if needed, and send it by email only after approval."
)
```

### 4.1 连接两个 MCP server

```python
NEWSPAPER_MCP_URL = "http://localhost:8080/mcp"
PREFERENCE_AGENT_MCP_URL = "http://localhost:8082/mcp"


def setup_news_agent_config(
    newspaper_url: str = NEWSPAPER_MCP_URL,
    preference_agent_url: str = PREFERENCE_AGENT_MCP_URL,
) -> Path:
    """创建 FastAgent 配置，让 News Agent 同时连接两个 server。"""
    temp_dir = Path(tempfile.mkdtemp(prefix="news_agent_"))
    config_path = temp_dir / "fastagent.config.yaml"

    config_content = f"""openai:
  base_url: "https://openrouter.ai/api/v1"

default_model: "openrouter.anthropic/claude-haiku-4.5"

mcp:
    servers:
        newspaper:
            transport: "http"
            url: "{newspaper_url}"
        preference_agent:
            transport: "http"
            url: "{preference_agent_url}"
"""

    config_path.write_text(config_content, encoding="utf-8")
    return config_path
```

### 4.2 News Agent 的工作流指令

```python
NEWS_AGENT_INSTRUCTION = """You are the user-facing News Generator Agent.

Required workflow:
1. Use newspaper tools to discover stories and create a newspaper draft.
2. Before publishing, preview or summarize the complete draft.
3. Call preference_agent.chat with the COMPLETE draft content.
4. If the response contains "DENIED", revise using the feedback, then review again.
5. Try at most 3 review/revision rounds.
6. If the response contains "APPROVED", call validate_and_finalize.
7. If validation fails, apply the suggested fixes and re-validate.
8. When approved and valid, call publish_newspaper with delivery_method="email" unless the user asks not to send.
9. After publishing, call preference_agent.chat again with delivery summary and user feedback so it can store useful preference patterns.

Important:
- Do not ask the user to call Preference Agent directly.
- Do not publish denied drafts.
- Do not publish invalid drafts.
- Include actual draft content when asking for review.
"""
```

### 4.3 定义 News Generator Agent

```python
news_agent_app = FastAgent(
    "News Agent Client",
    config_path=str(setup_news_agent_config()),
)


@news_agent_app.agent(
    name="News Generator",
    instruction=NEWS_AGENT_INSTRUCTION,
    servers=["newspaper", "preference_agent"],
    request_params=RequestParams(max_iterations=80),
)
async def news_generator():
    """用户面对的生成 Agent。"""
    pass


async def ask_news_agent(user_request: str) -> str:
    """用户通过这个函数和 News Agent 对话。"""
    async with news_agent_app.run() as agent:
        return await agent(user_request)
```

### 4.4 用户如何交互

用户只需要改这一句话：

```python
response = await ask_news_agent(
    "Create a short morning AI news brief for me. Review it against my preferences, revise if needed, and send it by email only after approval."
)

print(response)
```

---

## 5. 完整运行顺序

### 第一步：配置环境变量

```text
OPENROUTER_API_KEY
MCP_SMTP_FROM_EMAIL
MCP_SMTP_PASSWORD
```

### 第二步：启动 Newspaper MCP Server

运行：

```text
0202_advanced_mcp_smart_tools.ipynb
```

它会启动：

```text
http://localhost:8080/mcp
```

### 第三步：启动 Preference Agent Server

运行：

```text
0301_multi_agent_collaboration.ipynb
```

它会启动：

```text
http://localhost:8082/mcp
```

### 第四步：用户和 News Agent 交互

运行：

```text
0401_news_agent_client.ipynb
```

然后修改最后一个 cell 里的自然语言请求。

---

## 6. 这个项目是否实现了 RAG？

实现了，但它不是传统问答式 RAG，而是 Agentic RAG。

你的 RAG 链路是：

```text
文章/偏好写入 ChromaDB
  ↓
根据当前生成任务做语义检索
  ↓
把检索出的历史文章/用户偏好放入 prompt
  ↓
LLM 生成新闻摘要、审核建议或最终 newspaper
```

核心代码是：

```python
related = article_memory.search_articles(query=article["content"], limit=5)
```

然后：

```python
RELATED PAST COVERAGE:
{chr(10).join(f"- {r["title"]}" for r in related[:3])}
```

这就是“检索增强生成”。

---

## 7. 当前项目的边界

这个项目是一个可以展示完整 Agent 工作流的原型，但还不是完整产品化系统。

已经有：

- News Generator Agent。
- Preference Agent。
- MCP 工具通信。
- ChromaDB 记忆。
- RAG 式检索增强。
- 邮件发送。
- 审核通过/不通过逻辑。

还不够产品化的地方：

- 底层服务类如 `ArticleMemoryService`、`MemoryService` 需要完整源码和依赖。
- 主要运行方式仍然是 notebook。
- 没有网页 UI。
- 不通过后重新生成主要依赖 Agent 指令，而不是完全硬编码的 workflow controller。
- 上下文溢出主要靠 top-k 检索和截断预防，没有完整异常恢复机制。

