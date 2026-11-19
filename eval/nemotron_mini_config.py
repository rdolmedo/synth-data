"""Identical mini v2 settings across the bash and bash+editor ablation."""
import json
import os
import sys
from pathlib import Path

import yaml


def config(kind, base):
    editor = kind == "mini-swe-agent-2-editor"
    tools = ["bash", "file_editor"] if editor else ["bash"]
    baseline = yaml.safe_load((Path(__file__).parent / "configs/mini_v2.yaml").read_text())
    agent = baseline["agent"]
    if editor:
        agent["system_template"] += " You can also use a file editor."
        agent["instance_template"] = agent["instance_template"].replace(
            "AT LEAST ONE bash tool call", "AT LEAST ONE bash or file_editor tool call")
        agent["instance_template"] += (
            "\nUse file_editor to view, create, str_replace, insert, or undo_edit files. "
            "file_editor paths must be absolute.\n")
    # The fork's batch agent dispatches file_editor. Its interactive CLI subclass
    # overrides execute_actions and bypasses that dispatch, even in --yolo mode.
    agent.update(agent_class="default", step_limit=0, cost_limit=0)
    return {
        "agent": {
            **agent,
        },
        "environment": {**baseline["environment"], "environment_class": "local"},
        "model": {
            **baseline["model"],
            "model_class": "litellm", "tools": tools, "cost_tracking": "ignore_errors",
            "format_error_template": "Tool call error: {{error}}. Use at least one native "
                + " or ".join(tools) + " tool call. To finish, call bash with exactly: "
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
            "model_kwargs": {
                "api_base": base, "api_key": "EMPTY",
                "temperature": float(os.environ.get("TEMPERATURE", "1.0")),
                "top_p": float(os.environ.get("TOP_P") or "0.95"),
                "max_tokens": int(os.environ.get("MAX_TOKENS", "65536")),
                "timeout": 3600, "parallel_tool_calls": False,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        },
    }


if __name__ == "__main__":
    print(json.dumps(config(*sys.argv[1:])))
