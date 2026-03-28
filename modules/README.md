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
| `ctx.get_config(module_name)` | Return the current agent's config for the module (context-aware — automatically reads from the source agent during `on_output`, active agent during `on_user_input`) |
| `ctx.set_config(module_name, key, value)` | Write a single key into the active agent's module config |
| `ctx.request_action(command, agent, desc, action, on_deny)` | Request an interceptable action (see below) |

---

## Command interception

Modules can declare commands that are interceptable — meaning the user can choose to approve or deny them before they execute. This is useful for commands with side effects (e.g., routing a message to another agent).

### Declaring interceptable commands

Add an `INTERCEPT_SCHEMA` class attribute to your module:

```python
from params import ModuleParam, InterceptableCommand

class MyModule(Module):
    NAME = "my_module"
    INTERCEPT_SCHEMA = [
        InterceptableCommand("do_something", "Description of the action"),
    ]
```

Each `InterceptableCommand` has:
- `name` — the command identifier (used in config and `request_action`)
- `description` — shown in the Parameter Panel and module panel
- `intercept` — whether interception is on by default (`False` = auto-approve)

### Using `ctx.request_action()`

Instead of executing a command directly, call `request_action`:

```python
def _do_work(self, agent_name):
    desc = "About to do something important"

    def action():
        # This runs if approved (or if interception is off)
        self._ctx.emit("Did the thing!")

    def on_deny():
        # Optional — runs if the user clicks Deny
        self._ctx.emit("Action was blocked by user.")

    self._ctx.request_action("do_something", agent_name, desc,
                             action=action, on_deny=on_deny)
```

**Behavior:**
- If interception is **off** for this command+agent: `action()` fires immediately
- If interception is **on**: the request appears in the **Module Panel** (collapsible right panel) for the user to Allow or Deny

### Per-agent intercept config

Each agent has independent intercept settings in the **Parameter Panel** under `[module] <name> — intercepts`. The settings are persisted in the agent's JSON config under `module_intercept`:

```json
{
  "module_intercept": {
    "message_agent": {
      "route_message": true
    }
  }
}
```

### Module Panel

The module panel is a collapsible panel on the right side of the console, toggled via the ☰ drawer button in the console toolbar row (next to the Output/Basic Logs/Verbose Logs buttons). It shows pending interception requests as cards with Allow/Deny buttons. Features:

- **Badge counter** on the ☰ drawer button shows total pending count
- **Agent filter** — when an individual agent is selected, only that agent's requests are shown (with a note about hidden requests from other agents)
- **Auto-expand** — the panel expands automatically when a new request arrives
- **Max 50 pending** — oldest request is auto-denied when the cap is exceeded
- Requests are cleaned up when an agent is deleted or unloaded

### Known limitation

The `_route_count` rate limiter in `message_agent` is approximate with async approval. If a user sends a new prompt while old interception requests are pending, the count resets. The `do_route` closure re-validates at execution time to mitigate this.

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
- If interception is on: a request card appears in the Module Panel for user approval
- If interception is off (default): the message is routed immediately
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
