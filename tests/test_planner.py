import pytest

from self_evolution_agent.planner import Planner
from self_evolution_agent.schemas import (
    ExecutionPlan,
    IncomingMessage,
    Intent,
    PlannedTask,
    TaskKind,
)


def message(text: str = "", images: list | None = None) -> IncomingMessage:
    return IncomingMessage(
        event_id="event-1",
        message_id="message-1",
        chat_id="chat-1",
        open_id="user-1",
        text=text,
        images=images or [],
    )


def test_heuristic_routes_idea() -> None:
    plan = Planner.heuristic_plan(message("灵感：做一个自动整理书签的工具"))
    assert plan.intent == Intent.INSPIRATION
    assert plan.tasks[0].kind.value == "inspiration"


def test_heuristic_prioritizes_image_ingest() -> None:
    from self_evolution_agent.schemas import ImageAttachment

    plan = Planner.heuristic_plan(message("看看能做什么菜", [ImageAttachment(image_key="img-1")]))
    assert [task.intent for task in plan.tasks] == [Intent.FRIDGE_INGEST]
    assert plan.requires_confirmation is True


def test_unknown_routes_general_chat() -> None:
    plan = Planner.heuristic_plan(message("今天天气怎么样"))
    assert plan.intent == Intent.GENERAL_CHAT
    assert plan.tasks[0].kind == TaskKind.GENERAL


def test_heuristic_routes_recipe_and_used_up_food() -> None:
    assert Planner.heuristic_plan(message("今晚做什么菜")).intent == Intent.RECIPE
    mutation = Planner.heuristic_plan(message("牛奶用完了"))
    assert mutation.intent == Intent.FRIDGE_MUTATE
    assert mutation.requires_confirmation is True


def test_model_plan_rejects_multiple_tasks() -> None:
    plan = ExecutionPlan(
        intent=Intent.INSPIRATION,
        tasks=[
            PlannedTask(
                id="task-1",
                kind=TaskKind.INSPIRATION,
                intent=Intent.INSPIRATION,
                instruction="保存灵感",
            ),
            PlannedTask(
                id="task-2",
                kind=TaskKind.GENERAL,
                intent=Intent.GENERAL_CHAT,
                instruction="普通对话",
            ),
        ],
    )
    with pytest.raises(ValueError, match="exactly one"):
        Planner.validate_plan(plan, message("灵感：测试"))


def test_model_plan_is_rejected_when_mutation_skips_confirmation() -> None:
    plan = ExecutionPlan(
        intent=Intent.FRIDGE_MUTATE,
        tasks=[
            PlannedTask(
                id="task-1",
                kind=TaskKind.FRIDGE,
                intent=Intent.FRIDGE_MUTATE,
                instruction="牛奶用完了",
            )
        ],
    )
    with pytest.raises(ValueError, match="confirmation"):
        Planner.validate_plan(plan, message("牛奶用完了"))


def test_model_plan_rejects_dependencies_until_workflow_supports_them() -> None:
    plan = ExecutionPlan(
        intent=Intent.INSPIRATION,
        tasks=[
            PlannedTask(
                id="task-1",
                kind=TaskKind.INSPIRATION,
                intent=Intent.INSPIRATION,
                instruction="保存灵感",
                dependencies=["other-task"],
            )
        ],
    )
    with pytest.raises(ValueError, match="dependencies"):
        Planner.validate_plan(plan, message("灵感：测试"))


def test_model_plan_requires_fridge_ingest_for_image() -> None:
    from self_evolution_agent.schemas import ImageAttachment

    plan = ExecutionPlan(
        intent=Intent.INSPIRATION,
        tasks=[
            PlannedTask(
                id="task-1",
                kind=TaskKind.INSPIRATION,
                intent=Intent.INSPIRATION,
                instruction="保存图片说明",
            )
        ],
    )
    with pytest.raises(ValueError, match="fridge_ingest"):
        Planner.validate_plan(plan, message("", [ImageAttachment(image_key="img-1")]))
