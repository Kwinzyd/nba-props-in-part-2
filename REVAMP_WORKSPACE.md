# NBA pregame bot replacement

The implemented replacement is isolated at:

`C:\Users\louis\nba-props-in-v2`

Its [operating documentation](../nba-props-in-v2/README.md) and
[activation report](../nba-props-in-v2/ACTIVATION_2026-11-15.md) describe the
new modeling pipeline, prospective gate, tests, and rollback procedure.

The current bot continues running from this workspace. V2 initially runs in
shadow mode and takes over only after the agreed automatic validation gate.
The replacement has separate source, models, tables, logs, and supervision.
Historical data, existing uncommitted changes, and active production models
were preserved.
