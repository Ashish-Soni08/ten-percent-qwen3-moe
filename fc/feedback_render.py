import json
def render(calls):
    return "\n".join("<tool_call>\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False) + "\n</tool_call>" for c in calls)
