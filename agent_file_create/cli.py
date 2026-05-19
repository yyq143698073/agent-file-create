import logging
import uuid
from pathlib import Path

from agent_file_create.agent import DocumentAgent
from agent_file_create.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _pick_files(resource_dir: Path) -> list[str]:
    files = [p for p in resource_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
    files = sorted(files, key=lambda x: x.name.lower())
    if not files:
        return []
    print("resource目录：", str(resource_dir))
    print("可用文件：")
    for p in files:
        print(" -", p.name)
    print("请输入resource目录下的文件名（可输入多个，用空格/逗号分隔；输入q结束选择）：", end="")
    sel = input().strip()
    if not sel or sel.lower() == "q":
        return []
    names = [x.strip() for x in sel.replace(",", " ").split() if x.strip()]
    chosen: list[str] = []
    for n in names:
        p = resource_dir / n
        if p.exists() and p.is_file():
            chosen.append(str(p))
    return chosen


def main() -> None:
    setup_logging()
    base_dir = Path(__file__).resolve().parent.parent
    resource_dir = base_dir / "resource"
    try:
        resource_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        file_paths = _pick_files(resource_dir)
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        return

    if not file_paths:
        print("未选择文件。请将文件放入 resource 目录后重试。")
        return

    try:
        print("请输入自然语言需求（用于生成大纲与正文，留空使用默认）：", end="")
        user_prompt = input().strip() or "生成一份报告"
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        return

    task_id = uuid.uuid4().hex[:8]
    agent = DocumentAgent(task_id=task_id, user_prompt=user_prompt, file_paths=file_paths, template_dir_override=None)
    state = agent.run(human_input_fn=lambda q: input((q or "").strip() + "\n> "))
    out_dir = state.get("output_dir") or (base_dir / "result" / task_id)
    print("任务ID:", task_id)
    print("输出目录:", out_dir)
    print("已生成大纲与正文；如存在模板则已渲染输出。")

