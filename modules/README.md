# ClearHelm Modules

Drop any `.py` file here and it will be loaded automatically on startup. No configuration required.

---

## How it works

On startup, the app scans this directory for `*.py` files (alphabetical order, files starting with `_` are skipped). Each file is imported, its `MODULE_CLASS` attribute is read, and the class is instantiated and given a `ModuleContext`. From that point the module receives lifecycle callbacks as the app runs.

If a module fails to load (syntax error, missing import, exception in `on_load`), the error is logged to the terminal and the app continues — a bad module never crashes the app.

Module output (via `ctx.emit()`) appears in the **All** view under the `system` sender.

---

## Writing a module

Every module file must expose a `MODULE_CLASS` attribute pointing to a subclass of `Module`.

```python
from module_manager import Module, ModuleContext

class MyModule(Module):

    def on_load(self, ctx: ModuleContext):
        # Called once at startup. Store ctx if you need it later.
        ctx.emit("[my_module] Loaded.")

    def on_output(self, model_name: str, text: str):
        # Called on every text chunk streamed from an agent.
        # model_name is "system" for log lines — skip those if you only
        # care about real agent output.
        pass

    def on_user_input(self, text: str) -> bool:
        # Called before a user prompt is dispatched.
        # Return True to consume the input (clears the field, skips normal send).
        # Return False to let normal processing continue.
        return False

    def on_unload(self):
        # Called when the app is closing. Clean up resources here.
        pass

MODULE_CLASS = MyModule
```

All hooks are optional — only override the ones you need.

---

## ModuleContext API

| Method | Description |
|---|---|
| `ctx.emit(text)` | Write a line to the console (visible in All view, sender shown as `system`) |
| `ctx.message_agent(name, message)` | Send `message` to the named agent's prompt queue |
| `ctx.get_agent_names()` | Returns a list of all registered agent names |
| `ctx.get_ready_agents()` | Returns a list of agents currently in READY state |

---

## Included modules

### `message_agent.py` — inter-agent message routing

Enables agents to route messages to each other by outputting a toolcall tag, and provides a manual trigger for testing.

#### Toolcall detection

The module scans every agent's output stream for this exact pattern:

```
<toolcall>message_agent("TargetAgent", "your message here")</toolcall>
```

When matched:
- The target agent name is validated against the registered agent list
- If valid, the message is submitted to that agent's prompt queue
- If the agent name is unknown, a warning is logged and the tag is ignored

The tag can appear anywhere in the agent's output — before, after, or mixed with other text.

**Getting an untrained model to output a toolcall reliably:**

```
Output exactly the following text, with no other content before or after it:
<toolcall>message_agent("AgentName", "hello")</toolcall>
```

For ongoing use, add a system prompt to the agent's config that defines the format:

```
When you want to send a message to another agent, output:
<toolcall>message_agent("AgentName", "your message")</toolcall>
```

#### Loop prevention

A maximum of 10 routed messages are allowed per user action. If agents route to each other recursively and hit the limit, further routing is suppressed and a warning is logged. The counter resets each time the user sends a new prompt.

#### User trigger: `!test_route`

Type `!test_route` in the input field and press Enter. The input is consumed (no echo, no agent dispatch). The module finds the first READY agent and sends it a hardcoded test message, logging the result to the console.

Useful for confirming the routing pipeline works before testing with actual agent output.
