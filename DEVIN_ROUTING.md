# Devin → Mini Gateway Routing

Devin should normally call the DAWN capability interface rather than selecting a model vendor directly.

## Public MCP tools

- `mini_status` — installed/ready worker status
- `dawn_capabilities` — capability registry
- `dawn_plan_route(capability, allow_paid=false)` — dry-run route decision
- `dawn_execute(capability, task, workspace, approved=false, allow_paid=false)` — governed execution

Legacy explicit worker tools remain available for diagnostics and controlled overrides.

## Default policy

1. Local/free worker first.
2. Do not use paid/quota workers unless `allow_paid=true` was explicitly requested.
3. `content_publish` and `deploy` require `approved=true`.
4. After material work, use `verify` when independent evidence is required.
5. Never treat a tool's prose claim as proof of completion.

## Suggested Devin behaviour

For an ordinary software task:

1. call `dawn_plan_route`;
2. call `dawn_execute` with the relevant capability;
3. inspect structured execution evidence;
4. call `dawn_execute(capability="verify", ...)` for independent verification when appropriate;
5. only escalate to paid agents after a local route is unavailable or demonstrably inadequate and the user permits escalation.
