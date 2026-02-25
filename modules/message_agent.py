"""message_agent module — inter-agent message routing for ClearHelm.

Demonstrates two features:
1. Toolcall detection: scans agent output for
       <toolcall>message_agent("TargetAgent", "message")</toolcall>
   and routes the message to the named agent.

2. User trigger: typing ``!test_route`` sends a hardcoded test message to the
   first READY agent and consumes the input (no echo, no normal dispatch).
"""

import re

from module_manager import Module, ModuleContext

# Maximum chars kept per-agent in the rolling lookahead buffer.
_BUFFER_MAX = 4096

# Maximum routed messages triggered by a single user action before we stop.
# Prevents runaway agent-to-agent loops.
_MAX_ROUTES = 10

_TOOLCALL_RE = re.compile(
    r'<toolcall>\s*message_agent\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*</toolcall>',
    re.DOTALL,
)


class MessageAgentModule(Module):
    def on_load(self, ctx: ModuleContext):
        self._ctx = ctx
        self._buffers: dict[str, str] = {}
        self._route_count = 0
        ctx.emit("[message_agent] Module loaded.\n")

    def _route(self, source: str, target: str, message: str):
        if self._route_count >= _MAX_ROUTES:
            self._ctx.emit(
                f"[message_agent] Route limit ({_MAX_ROUTES}) reached — suppressed."
            )
            return
        self._route_count += 1
        self._ctx.emit(
            f"[message_agent] Routing from '{source}' → '{target}': {message!r}"
        )
        self._ctx.message_agent(target, message)

    def on_output(self, model_name: str, text: str):
        # Skip log lines emitted by ctx.emit() itself
        if model_name == "system":
            return

        buf = self._buffers.get(model_name, "") + text
        if len(buf) > _BUFFER_MAX:
            buf = buf[-_BUFFER_MAX:]

        offset = 0
        for m in _TOOLCALL_RE.finditer(buf):
            target, message = m.group(1), m.group(2)
            offset = m.end()  # always consume — even if target is unknown
            known = self._ctx.get_agent_names()
            if target not in known:
                self._ctx.emit(
                    f"[message_agent] Unknown agent '{target}' — ignored."
                )
                continue
            self._route(model_name, target, message)

        self._buffers[model_name] = buf[offset:]

    def on_user_input(self, text: str) -> bool:
        self._route_count = 0  # reset limit on each new user action
        if text.strip() != "!test_route":
            return False
        ready = self._ctx.get_ready_agents()
        if not ready:
            self._ctx.emit("[message_agent] !test_route: no READY agents available.")
            return True
        target = ready[0]
        test_msg = "Hello from the module system — this is a test routed message."
        self._ctx.emit(f"[message_agent] !test_route → '{target}': {test_msg!r}")
        self._ctx.message_agent(target, test_msg)
        return True

    def on_unload(self):
        self._buffers.clear()


MODULE_CLASS = MessageAgentModule
