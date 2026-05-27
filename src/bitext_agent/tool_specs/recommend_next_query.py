"""Versioned `recommend_next_query` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import RecommendationResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class RecommendNextQueryArgs(BaseModel):
    """Arguments for recommending a next query."""

    session_id: str = Field(description="Current session ID.")
    user_id: str = Field(description="Current user UUID or external user ID.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the recommend_next_query tool spec."""

    def recommend_next_query(session_id: str, user_id: str) -> RecommendationResult:
        turns = context.store.list_turns(session_id, limit=12)
        facts = context.store.list_profile_facts(context.user_uuid)
        text = " ".join(turn["content"].lower() for turn in turns)
        profile = " ".join(fact.fact.lower() for fact in facts)
        if "refund" in text or "refund" in profile or "money back" in text:
            query = "Show me 5 examples from the REFUND category."
            reason = "You have been looking at refund-related support records."
        elif "complaint" in text or "complaint" in profile:
            query = "Summarize how agents respond to complaint intents."
            reason = "Complaint response patterns are a useful follow-up to your recent questions."
        else:
            categories = context.repository.list_categories()
            category = categories[0] if categories else "ACCOUNT"
            query = f"What is the distribution of intents in the {category} category?"
            reason = "Intent distributions are a compact way to understand the dataset shape."
        context.store.set_pending_recommendation(session_id, context.user_uuid, query, reason)
        return RecommendationResult(query=query, reason=reason, pending=True)

    return ToolSpec(
        name="recommend_next_query",
        version="1.0.0",
        description="Suggest a context-aware next dataset query without executing it.",
        args_schema=RecommendNextQueryArgs,
        output_schema=RecommendationResult,
        callable=recommend_next_query,
        examples=[ToolExample(input={"session_id": "demo", "user_id": "demo"}, output_summary="Pending query.")],
        return_summary="Suggested query, reason, and pending flag.",
    )

