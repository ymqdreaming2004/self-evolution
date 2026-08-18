import hashlib
import json

from self_evolution_agent.api import parse_card_action, parse_message_event
from self_evolution_agent.config import Settings
from self_evolution_agent.long_connection import parse_long_connection_message
from self_evolution_agent.providers.feishu import FeishuClient, fridge_confirmation_card
from self_evolution_agent.schemas import IngredientPrediction, VisionResult


def test_parse_text_event() -> None:
    payload = {
        "header": {"event_type": "im.message.receive_v1", "event_id": "e1"},
        "event": {
            "sender": {"sender_id": {"open_id": "ou_1"}},
            "message": {
                "message_id": "m1",
                "chat_id": "c1",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "收藏 https://example.com/a"}),
            },
        },
    }
    message = parse_message_event(payload)
    assert message is not None
    assert message.open_id == "ou_1"
    assert str(message.urls[0]) == "https://example.com/a"


def test_parse_long_connection_text_event() -> None:
    class Value:
        def __init__(self, **values):
            self.__dict__.update(values)

    event = Value(
        header=Value(event_id="e1"),
        event=Value(
            sender=Value(sender_id=Value(open_id="ou_1")),
            message=Value(
                message_id="m1",
                chat_id="c1",
                chat_type="p2p",
                message_type="text",
                content=json.dumps({"text": "灵感：测试"}),
            ),
        ),
    )
    message = parse_long_connection_message(event)
    assert message is not None
    assert message.event_id == "e1"
    assert message.open_id == "ou_1"
    assert message.text == "灵感：测试"


def test_parse_card_form_values() -> None:
    payload = {
        "event": {
            "operator": {"operator_id": {"open_id": "ou_1"}},
            "action": {
                "value": {"action_id": "a1", "thread_id": "t1", "action": "confirm"},
                "form_value": {"name_0": "纯牛奶"},
            },
        }
    }
    action, open_id = parse_card_action(payload)
    assert open_id == "ou_1"
    assert action.values["name_0"] == "纯牛奶"


def test_confirmation_card_contains_editable_fields() -> None:
    result = VisionResult(
        items=[IngredientPrediction(name="牛奶", confidence=0.9)], model_version="v1"
    )
    card = fridge_confirmation_card(action_id="a1", thread_id="t1", draft_id="d1", result=result)
    assert card["schema"] == "2.0"
    form = card["body"]["elements"][1]
    assert form["tag"] == "form"
    names = [item.get("name") for item in form["elements"]]
    assert "name_0" in names
    submit = next(item for item in form["elements"] if item.get("tag") == "button")
    assert submit["form_action_type"] == "submit"
    assert not any(name and name.startswith("expiry_") for name in names)
    assert not any(name and name.startswith("quantity_") for name in names)


def test_feishu_signature_verification() -> None:
    settings = Settings(feishu_encrypt_key="secret")
    client = FeishuClient(settings)
    body = b'{"event":"test"}'
    signature = hashlib.sha256(b"123noncesecret" + body).hexdigest()
    assert client.verify_signature(body, "123", "nonce", signature)
