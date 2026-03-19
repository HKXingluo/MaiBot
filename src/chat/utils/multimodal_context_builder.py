import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from src.chat.utils.chat_message_builder import build_readable_messages_with_id, replace_user_references
from src.chat.utils.utils import is_bot_self
from src.chat.utils.utils_image import image_path_to_base64
from src.common.data_models.database_data_model import DatabaseMessages
from src.common.database.database_model import Images
from src.common.logger import get_logger
from src.llm_models.model_client.base_client import BaseClient
from src.llm_models.payload_content.message import Message, MessageBuilder, RoleType

logger = get_logger("chat")

_PICID_PATTERN = re.compile(r"\[picid:([^\]]+)\]")


@dataclass
class MultimodalContext:
    readable_context: str
    message_id_list: List[Tuple[str, DatabaseMessages]]
    multimodal_messages: List[Message]


def _guess_image_format(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in {"jpg", "jpeg", "png", "webp", "gif"}:
        return "jpeg" if ext == "jpg" else ext
    return "png"


def _build_message_with_picid(
    msg_id: str,
    msg: DatabaseMessages,
    support_formats: list[str],
    strict_image: bool,
) -> Message:
    builder = MessageBuilder()
    role = RoleType.Assistant if is_bot_self(msg.user_info.platform, msg.user_info.user_id) else RoleType.User
    builder.set_role(role)

    ts = f"{msg.time:.0f}" if msg.time else "0"
    sender = msg.user_info.user_nickname or msg.user_info.user_id or "未知用户"
    content = (msg.processed_plain_text or "").strip()
    content = replace_user_references(content, msg.user_info.platform or "qq", replace_bot_name=True)
    builder.add_text_content(f"[{msg_id}][{ts}] {sender}: ")

    last = 0
    for m in _PICID_PATTERN.finditer(content):
        if m.start() > last:
            builder.add_text_content(content[last : m.start()])
        pic_id = m.group(1)
        image = Images.get_or_none(Images.image_id == pic_id)
        if not image or not getattr(image, "path", ""):
            msg_text = f"[图片缺失:{pic_id}]"
            logger.error(f"构建多模态上下文失败，找不到图片记录: {pic_id}")
            if strict_image:
                raise ValueError(msg_text)
            builder.add_text_content(msg_text)
        else:
            image_b64 = image_path_to_base64(image.path)
            builder.add_image_content(
                image_format=_guess_image_format(image.path),
                image_base64=image_b64,
                support_formats=support_formats,
            )
        last = m.end()

    if last < len(content):
        builder.add_text_content(content[last:])
    if not content:
        builder.add_text_content("[空消息]")
    return builder.build()


def build_multimodal_context(
    messages: List[DatabaseMessages],
    timestamp_mode: str = "normal_no_YMD",
    read_mark: float = 0.0,
    truncate: bool = False,
    show_actions: bool = False,
    strict_image: bool = True,
) -> MultimodalContext:
    readable_context, message_id_list = build_readable_messages_with_id(
        messages=messages,
        timestamp_mode=timestamp_mode,
        read_mark=read_mark,
        truncate=truncate,
        show_actions=show_actions,
    )
    # 这里先放占位，真正支持格式在 message_factory 内由 client 提供
    multimodal_messages = []
    for msg_id, msg in message_id_list:
        mm_builder = MessageBuilder()
        mm_builder.add_text_content(f"[{msg_id}]")
        multimodal_messages.append(mm_builder.build())
    return MultimodalContext(
        readable_context=readable_context,
        message_id_list=message_id_list,
        multimodal_messages=multimodal_messages,
    )


def make_multimodal_messages_for_client(
    message_id_list: List[Tuple[str, DatabaseMessages]],
    client: BaseClient,
    strict_image: bool = True,
) -> List[Message]:
    support_formats = client.get_support_image_formats()
    return [_build_message_with_picid(msg_id, msg, support_formats, strict_image) for msg_id, msg in message_id_list]


def make_message_factory(
    system_prompt: str,
    message_id_list: List[Tuple[str, DatabaseMessages]],
    prefix_text: Optional[str] = None,
    strict_image: bool = True,
) -> Callable[[BaseClient], List[Message]]:
    def _factory(client: BaseClient) -> List[Message]:
        system_builder = MessageBuilder().set_role(RoleType.System).add_text_content(system_prompt)
        user_builder = MessageBuilder().set_role(RoleType.User)
        user_builder.add_text_content(prefix_text or "以下是聊天上下文：")
        user_msg = user_builder.build()
        multimodal_messages = make_multimodal_messages_for_client(message_id_list, client, strict_image=strict_image)
        return [system_builder.build(), user_msg, *multimodal_messages]

    return _factory
