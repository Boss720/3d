# Phase3.3 completion notices and keyboard access

Frozen scope: frontend only. Browser completion notifications are OFF at launch
and after every reload. Explicit button Enable completion notifications requests
permission only from that user action. Never request permission automatically.
Capability absence/denial/revocation yields actionable visible status; disabling
stops this tab's notices without changing browser-wide permission. No settings
preset/API/performance schema changes and no persistent preference in this slice.

Notify only ordinary exports observed first as uploading/queued/processing and
later completed, once per ID per tab. Existing terminal jobs discovered on initial
poll/refresh never notify, and clips never notify. Keep tracking bounded to512
IDs, avoid notification storms (one notice per poll, summarize count if needed).
No notification exception may break polling, merge, downloads or queue updates.
Title StereoLab export ready, text source filename or completed count; no paths
or diagnostics. Do not enable or grant real browser permission during automated QA.

Expose keyboard shortcut help near conversion controls: Alt+P generates the
current still preview and Alt+Enter submits ready queue. Use event.code with Alt
and no Ctrl/Meta/Shift. Ignore repeats, prevented/composing events, editable
input/textarea/select/contenteditable targets and open modal dialogs. Only invoke
the SAME existing enabled UI button click path. Prevent default only when action
is actually accepted. No shortcut cancels, removes, downloads or grants permission.
Buttons and disclosure keyboard focus remain native. Escape remains untouched.

Acceptance: red then green pure policy tests for opt-in/default/denial, terminal
baseline/clips/once-only/bounded tracking and shortcut target/modifiers/readiness;
independent frontend/full suites and browser keyboard check with generated media.
No backend, service restart, GPU invocation or actual browser permission grant.
