from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from .agents import FridgeAgent, InspirationAgent, PlaceholderAgent
from .effects import EffectExecutor, query_card_effect
from .planner import Planner
from .schemas import AgentEffect, AgentResult, ExecutionPlan, IncomingMessage, PlannedTask


class GraphState(TypedDict, total=False):
    message: dict[str, Any]
    thread_id: str
    plan: dict[str, Any]
    task: dict[str, Any]
    results: Annotated[list[dict[str, Any]], operator.add]
    effects: list[dict[str, Any]]
    reply: str
    pending_confirmation: dict[str, Any] | None
    confirmation_result: str


class AgentWorkflow:
    def __init__(
        self,
        *,
        planner: Planner,
        inspiration: InspirationAgent,
        fridge: FridgeAgent,
        placeholder: PlaceholderAgent,
        effects: EffectExecutor,
        checkpointer: Any,
    ):
        self.planner = planner
        self.inspiration = inspiration
        self.fridge = fridge
        self.placeholder = placeholder
        self.effects = effects
        self.graph = self._build(checkpointer)

    def _build(self, checkpointer: Any) -> Any:
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._plan)
        graph.add_node("inspiration", self._run_inspiration)
        graph.add_node("fridge", self._run_fridge)
        graph.add_node("placeholder", self._run_placeholder)
        graph.add_node("aggregate", self._aggregate)
        graph.add_node("execute_effects", self._execute_effects)
        graph.add_node("wait_confirmation", self._wait_confirmation)
        graph.add_node("apply_confirmation", self._apply_confirmation)
        graph.add_node("send_confirmation_result", self._send_confirmation_result)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", self._dispatch_tasks)
        graph.add_edge("inspiration", "aggregate")
        graph.add_edge("fridge", "aggregate")
        graph.add_edge("placeholder", "aggregate")
        graph.add_edge("aggregate", "execute_effects")
        graph.add_conditional_edges(
            "execute_effects",
            lambda state: "wait" if state.get("pending_confirmation") else "done",
            {"wait": "wait_confirmation", "done": END},
        )
        graph.add_edge("wait_confirmation", "apply_confirmation")
        graph.add_edge("apply_confirmation", "send_confirmation_result")
        graph.add_edge("send_confirmation_result", END)
        return graph.compile(checkpointer=checkpointer)

    async def _plan(self, state: GraphState) -> dict[str, Any]:
        message = IncomingMessage.model_validate(state["message"])
        plan = await self.planner.plan(message)
        return {"plan": plan.model_dump(mode="json"), "results": []}

    def _dispatch_tasks(self, state: GraphState) -> list[Send]:
        plan = ExecutionPlan.model_validate(state["plan"])
        return [
            Send(task.kind.value, {**state, "task": task.model_dump(mode="json")})
            for task in plan.tasks
        ]

    async def _run_inspiration(self, state: GraphState) -> dict[str, Any]:
        result = await self.inspiration.run(
            PlannedTask.model_validate(state["task"]),
            IncomingMessage.model_validate(state["message"]),
        )
        return {"results": [result.model_dump(mode="json")]}

    async def _run_fridge(self, state: GraphState) -> dict[str, Any]:
        result = await self.fridge.run(
            PlannedTask.model_validate(state["task"]),
            IncomingMessage.model_validate(state["message"]),
            state["thread_id"],
        )
        return {"results": [result.model_dump(mode="json")]}

    async def _run_placeholder(self, state: GraphState) -> dict[str, Any]:
        result = await self.placeholder.run(
            PlannedTask.model_validate(state["task"]),
            IncomingMessage.model_validate(state["message"]),
        )
        return {"results": [result.model_dump(mode="json")]}

    async def _aggregate(self, state: GraphState) -> dict[str, Any]:
        message = IncomingMessage.model_validate(state["message"])
        results = [AgentResult.model_validate(value) for value in state.get("results", [])]
        effects: list[AgentEffect] = []
        replies: list[str] = []
        for result in results:
            if result.error:
                replies.append(f"任务 {result.intent.value} 失败：{result.error}")
            elif result.reply:
                replies.append(result.reply)
            effects.extend(result.effects)
            if result.data.get("items") is not None:
                effects.append(
                    query_card_effect(
                        result.data, message_id=message.message_id, task_id=result.task_id
                    )
                )
        if replies:
            effects.append(
                AgentEffect(
                    type="send_text",
                    payload={"text": "\n\n".join(replies)},
                    idempotency_key=f"reply:{message.message_id}",
                )
            )
        return {
            "effects": [effect.model_dump(mode="json") for effect in effects],
            "reply": "\n\n".join(replies),
        }

    async def _execute_effects(self, state: GraphState) -> dict[str, Any]:
        message = IncomingMessage.model_validate(state["message"])
        pending = await self.effects.execute(
            [AgentEffect.model_validate(value) for value in state.get("effects", [])],
            open_id=message.open_id,
            thread_id=state["thread_id"],
        )
        return {"pending_confirmation": pending}

    async def _wait_confirmation(self, state: GraphState) -> dict[str, Any]:
        command = interrupt(state["pending_confirmation"])
        return {"pending_confirmation": command}

    async def _apply_confirmation(self, state: GraphState) -> dict[str, Any]:
        result = await self.effects.apply_confirmation(state["pending_confirmation"] or {})
        return {"confirmation_result": result}

    async def _send_confirmation_result(self, state: GraphState) -> dict[str, Any]:
        command = state.get("pending_confirmation") or {}
        open_id = command.get("open_id")
        if open_id:
            await self.effects.feishu.send_text(open_id, state["confirmation_result"])
        return {}

    async def invoke_message(self, message: IncomingMessage, thread_id: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.ainvoke(
            {"message": message.model_dump(mode="json"), "thread_id": thread_id, "results": []},
            config=config,
        )

    async def resume(self, thread_id: str, command: dict[str, Any]) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        return await self.graph.ainvoke(Command(resume=command), config=config)
