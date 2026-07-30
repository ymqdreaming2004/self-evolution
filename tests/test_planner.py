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


def test_heuristic_splits_image_and_recipe() -> None:
    from self_evolution_agent.schemas import ImageAttachment

    plan = Planner.heuristic_plan(message("看看能做什么菜", [ImageAttachment(image_key="img-1")]))
    assert {task.intent for task in plan.tasks} == {Intent.FRIDGE_INGEST, Intent.RECIPE}
    assert plan.requires_confirmation is True


def test_unknown_routes_placeholder() -> None:
    plan = Planner.heuristic_plan(message("今天天气怎么样"))
    assert plan.intent == Intent.PLACEHOLDER


def test_model_plan_is_rejected_when_mutation_skips_confirmation() -> None:
    plan = ExecutionPlan(
        intent=Intent.FRIDGE_MUTATE,
        tasks=[
            PlannedTask(
                id="task-1",
                kind=TaskKind.FRIDGE,
                intent=Intent.FRIDGE_MUTATE,
                instruction="删除库存项 abcdef12",
            )
        ],
    )
    with pytest.raises(ValueError, match="confirmation"):
        Planner.validate_plan(plan, message("删除 abcdef12"))


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
