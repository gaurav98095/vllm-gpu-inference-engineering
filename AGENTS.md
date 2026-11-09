# Repository instructions

These instructions apply whenever a new course module is written or an existing
module is substantially revised in this repository.

## Module artifacts are required

Every code-bearing module must include all three synchronized forms of the lab:

1. A complete Markdown lesson under `docs/modules/`.
2. An independent, runnable Python script under `code/module-XX/`.
3. A detailed, runnable Jupyter notebook under `code/module-XX/`.

The Python script and notebook must each contain enough setup and implementation
to run independently. Do not make the notebook only import a few functions from
the Python file, and do not make the Python file depend on notebook state. Shared
helpers are allowed, but both artifacts must remain usable on their own and must
document their assumptions, dependencies, inputs, measurements, and outputs.

## Code must be complete in all three places

When the lesson contains code, include the complete runnable code in:

- the Markdown lesson, in readable fenced code blocks;
- the Python script, with a CLI or clear runnable entry point where practical;
- the notebook, split into logical cells with comments and explanatory text.

Do not leave placeholder fragments, elided functions, unexplained variables, or
truncated snippets in any artifact. Code shown in the Markdown must match the
behavior of the companion files. If a benchmark requires GPU synchronization,
warmup, cache construction, timing methodology, or cleanup, show those details
explicitly.

## Notebooks must be detailed

Every notebook should be a complete guided lab, not just a code dump. Include:

- a setup and dependency cell;
- environment and GPU checks;
- model/data construction and configuration inspection;
- commented implementation cells for every experiment;
- correct timing and synchronization methodology;
- result tables and useful plots;
- interpretation cells explaining what the measurements mean;
- cleanup or memory-management notes where relevant.

Keep code cells executable in order from a fresh kernel. Add comments explaining
the purpose of non-obvious lines, especially CUDA events, warmups, cache state,
tensor shapes, device placement, and benchmark boundaries.

## Documentation needs sufficient visuals

Each module should contain a useful number of diagrams or images in
`docs/assets/module-XX/`, usually at least three for a substantial technical
module. Choose visuals that materially explain the lesson, such as:

- architecture or data-flow diagrams;
- tensor-shape transformations;
- memory or cache layouts;
- scheduling timelines;
- performance curves, bottleneck charts, or comparison diagrams.

Prefer repository-native SVG diagrams when the visual is explanatory or
architectural. Give every image a descriptive filename, include it in the
Markdown with meaningful alt text, and explain what the reader should notice.

## Module consistency

- Preserve the established module order and numbering.
- Update `README.md`, `docs/SUMMARY.md`, and `docs/index.md` when adding a module.
- Link each lesson to its `.py`, `.ipynb`, and visual assets.
- Keep the module title, progress count, companion filenames, and next-module
  links consistent everywhere.
- Do not add assignment, quiz, knowledge-check, or final-challenge sections;
  retain practical observations and interpretation prompts inside the lesson.
- Keep the lesson technically accurate and distinguish theoretical estimates
  from measured hardware results.

## Validation before handoff

Before committing a module:

- compile the Python script;
- validate notebook JSON and parse its code cells;
- validate SVG/XML assets;
- run `git diff --check`;
- verify that all Markdown links and companion files exist;
- inspect the final worktree so unrelated or parallel-agent changes are not
  staged accidentally.
