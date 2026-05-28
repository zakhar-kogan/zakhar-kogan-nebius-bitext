"""Versioned `get_user_profile` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import ProfileResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class GetUserProfileArgs(BaseModel):
    """Arguments for reading user profile memory."""

    user_id: str = Field(description="External or internal user identifier.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the get_user_profile tool spec."""

    def get_user_profile(user_id: str) -> ProfileResult:
        user_uuid = context.user_uuid if user_id in {context.user_uuid, "current"} else user_id
        return ProfileResult(user_uuid=user_uuid, facts=context.store.list_profile_facts(user_uuid))

    return ToolSpec(
        name="get_user_profile",
        version="1.0.0",
        description=(
            "Read distilled profile facts for the current user. Use this for explicit "
            "profile-memory questions such as 'what do you remember about me'."
        ),
        args_schema=GetUserProfileArgs,
        output_schema=ProfileResult,
        callable=get_user_profile,
        examples=[ToolExample(input={"user_id": "current"}, output_summary="Profile facts.")],
        return_summary="User UUID and active facts.",
    )
