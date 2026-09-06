# Baseline

The PTY harness ran the source CLI at widths 40, 80, and 120. It injected a
deterministic provider but used the production CLI, Rich UI, `read_file` tool,
and session persistence.

All three widths show the final assistant answer, but omit the actual tool result
(`fixture tool output`). At 40 columns the compact status row soft-wraps in the
middle of semantic fields, for example `ctx` and its numeric value split onto
separate terminal lines. The custom input border is also split between the
prompt line and the closing border. At normal width, the same status line relies
on a single long line and cannot explain a paused approval clearly.

The baseline approval capture is produced separately by the same harness. It
prints the raw JSON arguments but renders `allow [o]nce / [d]eny?` as
`allow nce / eny?`: Rich parses the bracketed letters as markup. The display
therefore hides every keyboard shortcut even though the input handler accepts
`o` and `d`. This capture was taken while the UI worktree was in progress; it
also confirms that the approval prompt was not yet wired to the new spinner
pause boundary.
