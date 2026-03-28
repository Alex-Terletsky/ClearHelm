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
from params import ModuleParam, InterceptableCommand

_TOOLCALL_RE = re.compile(
    r'<toolcall>\s*message_agent\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*</toolcall>',
    re.DOTALL,
)


class MessageAgentModule(Module):
    NAME = "message_agent"
    CONFIG_SCHEMA = [
        ModuleParam("buffer_max", int, 4096, "Max chars in lookahead buffer"),
        ModuleParam("max_routes", int, 10, "Max routed messages per user action"),
    ]
    INTERCEPT_SCHEMA = [
        InterceptableCommand("route_message", "Route a message to another agent", intercept=True),
    ]

    def on_load(self, ctx: ModuleContext):
        self._ctx = ctx
        self._buffers: dict[str, str] = {}
        self._route_count = 0
        ctx.emit("[message_agent] Module loaded.\n")

    def _get_cfg(self) -> dict:
        return self._ctx.get_config("message_agent")

    def _route(self, source: str, target: str, message: str):
        max_routes = self._get_cfg().get("max_routes", 10)
        if self._route_count >= max_routes:
            self._ctx.emit(
                f"[message_agent] Route limit ({max_routes}) reached — suppressed."
            )
            return

        desc = f"Route from '{source}' to '{target}': {message!r}"

        def do_route():
            # Re-validate at execution time (state may have changed since request)
            if self._route_count >= self._get_cfg().get("max_routes", 10):
                self._ctx.emit("[message_agent] Route limit reached — blocked.")
                return
            self._route_count += 1
            self._ctx.emit(
                f"[message_agent] Routing from '{source}' → '{target}': {message!r}"
            )
            self._ctx.message_agent(target, message)

        self._ctx.request_action(
            "route_message", source, desc,
            action=do_route,
            on_deny=lambda: self._ctx.emit("[message_agent] Route blocked by user."))

    def on_output(self, model_name: str, text: str):
        # Skip log lines emitted by ctx.emit() itself
        if model_name == "system":
            return

        buffer_max = self._get_cfg().get("buffer_max", 4096)
        buf = self._buffers.get(model_name, "") + text
        if len(buf) > buffer_max:
            buf = buf[-buffer_max:]

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
