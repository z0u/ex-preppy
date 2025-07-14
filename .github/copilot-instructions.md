We are writing code to run AI experiments. Most of the code is in Python.

### Development Flow

- Lint: `./go lint --fix` (ruff check)
- Format: `./go format` (ruff format)
- Typecheck: `./go types` (basedpyright)
- Test: `./go test` (pytest)
- Full CI check: `./go check` (includes build, fmt, lint, test)

Run `./go -h` to discover other commands.

### Repository structure

- `docs/**/*.ipynb`: Experiments, as Jupyter notebooks
- `publications/*.md`: Articles resulting from our experiments
- `src/**/*.py`: Reusable code
- `tests/**/*.py`
- `go`: Entrypoint for scripts (bash)
- `scripts/`: Linting, formatting, etc.

---

## Code-writing (generation)

### Style

Use JavaScript-style method chaining (newline before the dot, use outer parentheses as necessary).
Use cutting-edge syntax.
Prefer brevity.
Use single quotes for strings, except for multiline strings.

### Docstrings

Use the imperative for the first line of function docstrings.

```diff
- """Adds two numbers together"""
+ """Add two numbers together."""
```

### Typing

Use type hints.
Use `T | None` instead of `Optional[T]`.

```diff
- foo: Optional[int] = None
+ foo: int | None = None
```

### Modal

Use Modal-compatible patterns for distributed processing.
Returning a model from a remote training function may be infeasible if the model is large.
Most objects including custom functions and classes can be pickled and executed remotely.
Closures work even for remote functions, but don't _assume_ global scope.

### Notebooks

When working on a notebook, iterate on both the code (Python) and the prose (Markdown). Aim for a literate programming style in which we tell stories about our experiments. We don't just document the code; the notebook as a whole should display a strong narrative.

Put imports in the cell they're used in — or even in the function, if the module isn't used in the function's API. It makes re-running cells easier during development.

### Markdown and prose

Use sentence case for headings and descriptive lists.

```patch
- # Experiment Design
-   - **Foo Bar:** baz
+ # Experiment design
+   - **Foo bar:** baz
```

**Technical precision with accessibility:** Explain complex concepts clearly, with minimal jargon, but without oversimplifying. Bridge technical depth with broader implications for thoughtful readers.

**Structured analytical thinking:** Organize ideas hierarchically and build arguments systematically with clear delineation of concepts. Prefer paragraphs for nuanced or complex explanations, but use lists for summarizing steps, strengths, or when clarity would benefit from structure.

**Confident intellectual honesty:** Express uncertainty while maintaining substantive positions. Don’t be afraid to “think out loud” or use meta-commentary to explain your reasoning or process.

> "This blog is supposed to be a place where I can write about my understanding without worrying too much about accuracy. That said, I think there's a decent chance that I'm still missing something about superposition."

**Direct, conversational tone:** Use a friendly, approachable style that invites engagement. Avoid overly formal language, jargon-heavy explanations. Especially avoid sales pitch grandeur. Maintain a tone that is approachable and professional — informal enough to invite engagement, but always clear and technically precise.

> "If we can demonstrate reliable control over a model's internal representation with color, it suggests we might achieve similar control in more complex domains."

**Methodical documentation:** Be comprehensive yet concise, capturing both facts and context efficiently.

**Sentence structure:** Use varied sentence lengths: longer explanatory sentences broken up by shorter, punchy statements.

### Tools

This project uses `uv` for package management. Dependency groups: `dev`, `local`; see `pyproject.toml` for others. By default, PyTorch GPU dependencies are not installed.

When possible, use the built-in IDE tools rather than CLI commands:

- Use `get_errors` to check files for type and lint errors
- Use `insert_edit_into_file` to make changes (preserving unchanged code with `// ...existing code...`)
- use `run_tests` to run tests

Under the hood, these tools use ruff (formatting/linting), basedpyright (type-checking), and pytest (testing).

### Writing tests

This project uses pytest.

Use pytest idioms: fixtures, parametrize, assert.
Prefer functional tests (not class-based).
Prefer brevity where possible.

Prefer pytest-native tools.

```diff
+ from pytest import approx

- np.isclose(x, y)  # ❌
+ pytest.approx(x) == y  # ✅
```

Use structural assertions.

```diff
+ from unittest.mock import ANY

- assert 'x' in props and approx(props['x']) == 1.0  # ❌
- assert 'z' in props and approx(props['z']) == 0.8  # ❌
+ assert approx(props) == {'x': 1.0, 'y': ANY, 'z': 0.8}  # ✅
```

Use reserved domains to avoid accidentally fetching from real domains: `.example`, `.test`, `.invalid`.

```diff
- response = requests.get('test.com')  # ❌ this is a real domain!
+ response = requests.get('service.test')  # ✅ guaranteed not to resolve
```

### What to test?

**We only write valuable tests.** We test for behavioral verification under uncertainty:

- Exercise meaningful state transitions and invariants: Tests that verify your system maintains its promises; Boundary condition handling; State consistency across operations (e.g., after a series of mutations, derived state still makes sense)
- Capture domain logic and business rules: Scenarios that encode actual user workflows or data processing pipelines; Edge cases that reflect real-world complexity your system needs to handle
- Reveal integration assumptions: How your code behaves when dependencies return unexpected (but valid) responses; Error propagation and recovery behavior; Resource cleanup and lifecycle management
- Executable documentation: Tests that demonstrate intended usage patterns, with clear naming that explains the "given/when/then" story

Valuable tests fail _for interesting reasons_ — they break when you've actually broken something that matters to users. We rely on linters and type-checkers for everything else.
