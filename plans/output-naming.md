# Output naming templates — bounded Phase3.3 slice

Frozen contract after a130916. Existing per-job full UUID suffix and exclusive
output reservation remain mandatory. No source retention or automatic submission.

Template length counts Unicode codepoints in both backend and browser; the native
input may allow up to240 UTF16 units, with semantic validation enforcing120.
Reject all Unicode Cc/Cf/Cs literal characters consistently, including C1 controls,
directional/zero-width format characters and lone surrogates; astral pairs are valid.

Form /api/jobs adds output_name_template default '{name}_{layout}' for ordinary
exports. Processed clips retain their current internal preview names. Validate
before upload/model work: nonempty string <=120chars, only {name},{layout},{model},
{date}; no unmatched braces/format specs/conversions/attribute lookups, path
separators, control chars or Windows-invalid literal filename characters. Literals
may include spaces/dashes/dots/underscores and Unicode; rendered names sanitize
as existing ASCII filenames do. Date is job.created_at UTC YYYYMMDD (stable).
Render source stem, lowercase layout, model ID, date; sanitize entire rendered
stem, bound it to100characters, fall back to 'stereo-output' if empty, then append
FULL unique job UUID and automatic png/mp4 extension. Never expose user path parts.

Status naming_capabilities: schema_version1,supportedtrue,max_template_length120,
fields:['name','layout','model','date'],unique_job_id_suffix:true. Separate from
queue/performance schemas. Public job naming_schema_version1,
output_name_template, planned_output_name; actual output_name remains authoritative
once completed. Settings snapshot stays the exact existing11fields.

POST/api/jobs/{id}/naming JSON{output_name_template:string} only accepts pending
ordinary exports, using same atomic queue membership check and queue-before-jobs
locks. Invalid400, unknown404, active/dequeued/terminal/clip409. No partial mutation.
Changing queued layout/model updates planned filename but not created date/identity.

UI output template text field and example with supported tokens/maxlength/help;
capability-gated. Omit form field on older backend and keep older default behavior.
New ready items use current naming template at submit unless explicitly frozen;
per-file/batch Freeze settings freezes naming too. Retry uses original snapshot
naming if a browser File remains; duplicate uses current naming, both ready only.
Use settings loads valid naming snapshot too. Queue row shows template and planned
or actual name. Pending server export has separate Use current name action calling
naming endpoint; settings endpoint remains unchanged (avoids split-transaction race).
Naming changes do not invalidate visual previews/calibration or affect clip names.

Acceptance: strict template parsing, stable UTC date, basename sanitization,
full-ID collision protection/length, pre-upload rejection, queued naming atomicity,
legacy API/default filenames, per-file frozen naming, retry/duplicate contracts,
stale old-backend capability behavior. Independently run both suites and live image
output/name check. No EXE rebuild in this wave.
