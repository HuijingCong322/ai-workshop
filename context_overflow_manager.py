"""
Context overflow handling utilities for the AI newspaper agent.

这个文件补上原项目里缺少的“上下文窗口溢出恢复机制”：

1. 先做预防：限制文章正文、历史相关文章、用户偏好的长度。
2. 再做恢复：如果 LLM 调用仍然因为上下文太长失败，就自动压缩上下文并重试。
3. 最后做降级：多次失败后只保留最核心的信息，返回可控结果，而不是让 Agent 无限重试。

这些函数可以被 `0202_advanced_mcp_smart_tools.ipynb` 里的 `add_content_cluster()`
或其他调用 `ctx.sample(...)` 的工具复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable


CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "token limit",
    "too many tokens",
    "prompt is too long",
    "context window",
    "input is too long",
)


def estimate_tokens(text: str) -> int:
    """
    粗略估算 token 数。

    英文通常可以用 4 个字符约等于 1 token 粗估；
    中文一个字可能接近 1 token。这里取一个偏保守的估计，方便提前压缩。
    """
    if not text:
        return 0

    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return chinese_chars + max(1, other_chars // 4)


def trim_text(text: str, max_chars: int, suffix: str = "\n...[truncated]") -> str:
    """把长文本裁剪到指定字符数。"""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


def is_context_overflow_error(error: BaseException) -> bool:
    """判断异常是否像上下文窗口溢出。"""
    message = str(error).lower()
    return any(marker in message for marker in CONTEXT_OVERFLOW_MARKERS)


@dataclass
class ArticleContext:
    """生成单篇文章摘要所需的上下文。"""

    title: str
    content: str
    interests: list[str]
    related_titles: list[str]
    treatment: str = "brief"


def compact_article_context(context: ArticleContext, level: int) -> ArticleContext:
    """
    按压缩等级缩减上下文。

    level=0: 正常长度
    level=1: 减少正文和 related 数量
    level=2: 只保留核心正文和少量兴趣
    level=3: 极限降级，只保留标题、短正文和最核心偏好
    """
    content_limits = [3000, 1800, 900, 450]
    related_limits = [3, 2, 1, 0]
    interest_limits = [8, 5, 3, 2]

    index = min(level, len(content_limits) - 1)

    return ArticleContext(
        title=context.title,
        content=trim_text(context.content, content_limits[index]),
        interests=context.interests[: interest_limits[index]],
        related_titles=context.related_titles[: related_limits[index]],
        treatment=context.treatment,
    )


def build_article_summary_prompt(context: ArticleContext) -> str:
    """把压缩后的上下文组装成 LLM prompt。"""
    interests = "\n".join(f"- {topic}" for topic in context.interests) or "- No stored interests"
    related = "\n".join(f"- {title}" for title in context.related_titles) or "- No related past coverage"

    return f"""Summarize this article in {context.treatment} style.

USER INTERESTS:
{interests}

ARTICLE:
Title: {context.title}
Content: {context.content}

RELATED PAST COVERAGE:
{related}

Write a clear, useful summary for a personalized newspaper.
"""


async def safe_sample_with_context_retry(
    ctx: Any,
    base_context: ArticleContext,
    *,
    max_retries: int = 3,
) -> str:
    """
    安全调用 `ctx.sample(...)`，遇到上下文过长时自动压缩并重试。

    使用方式：

    ```python
    summary = await safe_sample_with_context_retry(
        ctx,
        ArticleContext(
            title=article["title"],
            content=article["content"],
            interests=interests.get("topics", []),
            related_titles=[r["title"] for r in related],
            treatment=treatment,
        ),
    )
    ```
    """
    last_error: BaseException | None = None

    for level in range(max_retries + 1):
        compacted = compact_article_context(base_context, level)
        prompt = build_article_summary_prompt(compacted)

        try:
            await ctx.debug(
                f"Sampling with context level={level}, estimated_tokens={estimate_tokens(prompt)}"
            )
            result = await ctx.sample(
                messages=[
                    {
                        "role": "user",
                        "content": {"type": "text", "text": prompt},
                    }
                ]
            )
            return result.text
        except Exception as error:
            last_error = error

            if not is_context_overflow_error(error):
                raise

            await ctx.warning(
                f"Context overflow at compression level {level}; retrying with smaller context."
            )

    fallback = compact_article_context(base_context, max_retries)
    return (
        f"{fallback.title}\n\n"
        f"Summary could not be generated with full context because the context window was exceeded. "
        f"Use this compact source excerpt instead:\n\n{fallback.content}"
    )


def select_top_related_titles(related: Iterable[dict[str, Any]], limit: int = 3) -> list[str]:
    """
    从 ChromaDB 检索结果里提取最重要的 related title。

    如果结果里有 similarity 字段，就按 similarity 从高到低排序；
    否则保持原顺序。
    """
    items = list(related)
    items.sort(key=lambda item: item.get("similarity", 0), reverse=True)
    return [item.get("title", "Untitled") for item in items[:limit]]

