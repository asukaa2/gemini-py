# Skills

Skills are user-defined markdown prompt files the agent can activate.
They turn into specialized roles / instructions the model can pull in
on demand — like gemini-cli's "agent skills".

## What is a skill?

A skill is a directory containing a `SKILL.md` file. The first markdown
paragraph becomes the skill's description (shown in `skills list`);
the rest of the file is the full prompt content (injected when the
skill is activated).

```
~/.gemini-py/skills/
└── code-reviewer/
    └── SKILL.md
```

`SKILL.md` example:

```markdown
# Code Reviewer

You are a meticulous code reviewer. Look for bugs, security issues,
and style violations. Always cite the file and line number when
pointing out issues.

## Checklist

- Input validation
- Error handling
- SQL injection / XSS
- Test coverage
```

## Discovery locations

Skills are discovered from (later wins on conflict):

1. `~/.gemini-py/skills/<name>/SKILL.md` (user-wide)
2. `./.gemini/skills/<name>/SKILL.md` (project-scoped, version-controlled)

You can put skills anywhere — just symlink or copy them into one of
these locations.

## Enabling / disabling skills

Skills are enabled by default when discovered. You can toggle them
without deleting the files:

```bash
gemini-py skills list              # see all + state
gemini-py skills enable code-reviewer
gemini-py skills disable code-reviewer
gemini-py skills enable-all
gemini-py skills disable-all
```

The state is persisted to `~/.gemini-py/skills_state.json`.

## Activating skills

### Inline with `@skill_name`

Any `@skill_name` token in your prompt is replaced inline with the
skill's full content wrapped in `--- BEGIN SKILL: <name> ---` markers:

```bash
gemini-py -p "@code-reviewer Please review src/foo.py"
```

After expansion, the prompt becomes:

```
Please
--- BEGIN SKILL: code-reviewer ---
# Code Reviewer
You are a meticulous code reviewer. ...
--- END SKILL: code-reviewer ---
 Please review src/foo.py
```

Unknown `@skill_name` tokens (or disabled skills) are left untouched
so the model can still pick them up as plain text mentions.

### Subagent delegation

When at least one skill is enabled, the `delegate_subagent(skill, task)`
tool is exposed to the model. The model can spawn a focused sub-agent
that runs its own agent turn with the skill's content as the system
prompt:

```bash
# Let the model decide to delegate
gemini-py -p "Review src/foo.py using the code-reviewer skill"
```

The sub-agent runs in its own `run_agent_turn()` call with:
- The skill's content as `system_instruction`
- The standard built-in tools (configurable)
- A reduced `max_iterations` (default 6 to keep it focused)

It returns its final text answer back to the parent agent, which can
then synthesize it into its own final answer.

### Disable skills / subagents

```bash
# Disable skill @-expansion + listing entirely
gemini-py -p "..." --no-skills

# Disable the delegate_subagent tool only (skills still listed)
gemini-py -p "..." --no-subagent
```

## Writing a good SKILL.md

A SKILL.md is just a markdown document. Here are some conventions that
work well:

### Include a clear role line

```markdown
# Code Reviewer

You are a meticulous code reviewer.
```

### Use a checklist

```markdown
## Checklist

- Input validation
- Error handling
- SQL injection / XSS
- Test coverage
```

### Be specific about output format

```markdown
## Output format

For each issue, output:

- **File**: path/to/file.py:42
- **Severity**: high / medium / low
- **Issue**: description
- **Fix**: suggested fix
```

### Include examples

```markdown
## Example

Input: `def add(a, b): return a + b`
Output:
- **File**: app.py:1
- **Severity**: low
- **Issue**: No type hints
- **Fix**: `def add(a: int, b: int) -> int: return a + b`
```

## Skills + subagent tool interplay

When the `delegate_subagent` tool is exposed (default), the model can
decide for itself when to delegate. For example, given the prompt
"Review src/foo.py", the model might:

1. Read `src/foo.py` itself (using the built-in `read_file` tool).
2. Decide to delegate the review to the `code-reviewer` skill.
3. Call `delegate_subagent(skill="code-reviewer", task="Review the
   file at src/foo.py for bugs and security issues")`.
4. The sub-agent runs its own turn (with `read_file` access), reads
   the file, applies the skill's checklist, and returns its findings.
5. The parent agent gets the sub-agent's findings as the tool result.
6. The parent agent synthesizes a final answer for the user.

This is more expensive (one parent turn + one child turn) but
produces higher-quality output for specialized tasks.

## Programmatic API

```python
from gemini_py.skills import SkillsLoader, expand_skills_in_prompt
from gemini_py.skills.subagent import delegate_to_subagent
from gemini_py import GeminiClient

# Load skills
loader = SkillsLoader()
print([s.name for s in loader.list()])
# ['code-reviewer', 'test-writer']

# Get a skill
skill = loader.get("code-reviewer")
print(skill.description)
print(skill.full_prompt())

# Expand @-tokens in a prompt
expanded = expand_skills_in_prompt("@code-reviewer review src/foo.py", loader)

# Direct subagent invocation
client = GeminiClient()
result = delegate_to_subagent(
    parent_client=client,
    skill_name="code-reviewer",
    task="Review src/foo.py",
    max_iterations=6,
)
print(result.text)
print(result.success, result.error)
```

## Skills + auto-memory interplay

Skills and `GEMINI.md` auto-memory work together:

- `GEMINI.md` provides persistent context (project conventions, rules).
- Skills provide role-specific instructions (a code-reviewer's checklist).

Both are injected into the system instruction at the start of every
agent turn. They don't conflict — `GEMINI.md` becomes "additional
context" at the bottom, while skills become an "available skills"
listing and the activated skill's content gets expanded into the
prompt.

## Example skills

See the upstream gemini-cli's `.gemini/skills/` for inspiration:
<https://github.com/google-gemini/gemini-cli/tree/main/.gemini/skills>

Some good starter skills:

- `code-reviewer` — review code for bugs, security, style
- `test-writer` — generate pytest tests
- `docs-writer` — write README / API docs
- `pr-creator` — write a git commit message + PR description
- `bug-fixer` — diagnose and fix a bug

## Migration from gemini-cli skills

If you have existing gemini-cli skills (TypeScript .md format), they
should work as-is — just copy the directory into
`~/.gemini-py/skills/`. The SKILL.md format is the same.

The only difference: gemini-cli uses `policy.toml` for advanced
control (sandboxing, tool whitelist) which we don't support yet.
