# Monorepo

This is a monorepo containing multiple independent projects.

## Projects

| Directory | Description |
|-----------|-------------|
| `A2A/` | A2A protocol work |
| `Agentic/` | Agentic systems |
| `causality_graph/` | Causality graph tooling |
| `chat_front/` | Chat frontend |
| `DA/` | Data analytics |
| `doc_logic_fsm/` | Document logic FSM |
| `graph_3gpp/` | 3GPP graph work |
| `graph_excel/` | Excel graph tooling |
| `graph_pdf/` | PDF graph tooling |
| `scm/` | SCM tooling |
| `spar/` | SPAR project |
| `transfer_learning/` | Transfer learning experiments |
| `yang_param/` | YANG parameter tooling |

## File Placement Rules (CRITICAL for agents)

- When working on a specific project, ALL files must be created inside that project's directory.
- Never write files to the repo root unless explicitly instructed.
- The repo root is NOT a project workspace — it is a container for projects.
- If unsure which project a file belongs to, ask before writing.

## Worktree + Subagent Rules

- When a worktree is created, it mirrors this same structure.
- Working path example: `/path/to/worktree/yang_param/` — not `/path/to/worktree/`.
- Always confirm the target project subdirectory before creating or editing files.
