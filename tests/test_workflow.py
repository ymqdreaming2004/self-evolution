from langgraph.checkpoint.memory import MemorySaver

from self_evolution_agent.schemas import (
    AgentEffect,
    AgentResult,
    ExecutionPlan,
    IncomingMessage,
    Intent,
    PlannedTask,
    TaskKind,
)
from self_evolution_agent.workflow import AgentWorkflow


class FakePlanner:
    async def plan(self, message):
        return ExecutionPlan(
            intent=Intent.INSPIRATION,
            tasks=[
                PlannedTask(
                    id="idea",
                    kind=TaskKind.INSPIRATION,
                    intent=Intent.INSPIRATION,
                    instruction=message.text,
                ),
                PlannedTask(
                    id="fallback",
                    kind=TaskKind.PLACEHOLDER,
                    intent=Intent.PLACEHOLDER,
                    instruction=message.text,
                ),
            ],
        )


class FakeAgent:
    def __init__(self, reply: str, confirmation: bool = False):
        self.reply = reply
        self.confirmation = confirmation

    async def run(self, task, message, thread_id=None):
        effects = []
        if self.confirmation:
            effects.append(
                AgentEffect(
                    type="inventory_mutation",
                    payload={"action": "consume", "item_id": "item-1"},
                    idempotency_key="confirm-1",
                    requires_confirmation=True,
                )
            )
        return AgentResult(task_id=task.id, intent=task.intent, reply=self.reply, effects=effects)


class FakeFeishu:
    def __init__(self):
        self.messages = []

    async def send_text(self, open_id, text):
        self.messages.append((open_id, text))


class FakeEffects:
    def __init__(self, pending=False):
        self.pending = pending
        self.feishu = FakeFeishu()

    async def execute(self, effects, *, open_id, thread_id):
        if self.pending and any(effect.requires_confirmation for effect in effects):
            return {"actions": [{"action_id": "a1"}], "open_id": open_id}
        return None

    async def apply_confirmation(self, command):
        return f"confirmed:{command['action_id']}"


def incoming() -> IncomingMessage:
    return IncomingMessage(
        event_id="e1",
        message_id="m1",
        chat_id="c1",
        open_id="u1",
        text="test",
    )


async def test_graph_parallel_dispatch_and_aggregate() -> None:
    effects = FakeEffects()
    workflow = AgentWorkflow(
        planner=FakePlanner(),
        inspiration=FakeAgent("idea"),
        fridge=FakeAgent("fridge"),
        placeholder=FakeAgent("fallback"),
        effects=effects,
        checkpointer=MemorySaver(),
    )
    result = await workflow.invoke_message(incoming(), "m1")
    assert {item["task_id"] for item in result["results"]} == {"idea", "fallback"}
    assert "idea" in result["reply"]
    assert "fallback" in result["reply"]


async def test_graph_interrupt_and_resume() -> None:
    planner = FakePlanner()
    planner.plan = lambda message: _confirmation_plan(message)
    effects = FakeEffects(pending=True)
    workflow = AgentWorkflow(
        planner=planner,
        inspiration=FakeAgent("idea", confirmation=True),
        fridge=FakeAgent("fridge"),
        placeholder=FakeAgent("fallback"),
        effects=effects,
        checkpointer=MemorySaver(),
    )
    paused = await workflow.invoke_message(incoming(), "confirm-thread")
    assert paused["__interrupt__"]
    resumed = await workflow.resume(
        "confirm-thread",
        {"action_id": "a1", "action": "confirm", "open_id": "u1", "values": {}},
    )
    assert resumed["confirmation_result"] == "confirmed:a1"
    assert effects.feishu.messages[-1] == ("u1", "confirmed:a1")


async def _confirmation_plan(message):
    return ExecutionPlan(
        intent=Intent.INSPIRATION,
        requires_confirmation=True,
        tasks=[
            PlannedTask(
                id="idea",
                kind=TaskKind.INSPIRATION,
                intent=Intent.INSPIRATION,
                instruction=message.text,
                requires_confirmation=True,
            )
        ],
    )
