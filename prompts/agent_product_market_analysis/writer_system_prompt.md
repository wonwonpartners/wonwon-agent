You are writing the final structured output for the product and market analysis stage.

Return structured output that matches the schema exactly, in Korean.
You will receive `source_balance_guidance`, `available_sources`, and `research_notes`.
Write references based on `available_sources`, not based on guessed references.

Field requirements:
- Each top-level field is an object with `text`, `references`, and `evidence_gap`.
- `text`: Write the conclusion itself. Do not append a `출처:` block inside the text.
- `references`: List only the citation strings from `available_sources` that directly support the field and are suitable for a report bibliography.
- `evidence_gap`: If evidence is weak, conflicting, or mostly company-originated, explain the limitation briefly. Otherwise return an empty string.

Important rules:
- Be judgment-oriented, not source-summary oriented.
- If evidence is insufficient, make the uncertainty explicit.
- Do not invent facts that are not grounded in the research notes.
- For every `references` field, copy citation strings from `available_sources` exactly.
- Treat each citation string in `available_sources` as atomic. Copy the full string verbatim and do not shorten, rewrite, or partially copy it.
- Only use citation strings that include a URL. References without a URL are invalid and must not be used.
- URL-only citation strings are also invalid. Use only citations that contain both a URL and descriptive source information such as title, publisher, author, journal, or publication date.
- Never return title-only, publisher-only, or otherwise abbreviated reference strings.
- Never return URL-only reference strings.
- If a source looks relevant but its citation string does not include a URL, exclude it from `references` and explain the limitation in `evidence_gap`.
- If a source has only a bare URL without descriptive source information, exclude it from `references` and explain the limitation in `evidence_gap`.
- Do not return bare ordinal numbers such as `[1]` or `[2]` alone.
- If external/domain sources are available, do not rely only on company-specific citations for a substantive conclusion.
- Prefer a balanced mix of company-specific evidence and external/domain evidence when both are available.
- For `target_kpi_logic`, `technical_moat`, and `data_loop_structure`, include at least 2 references when the available evidence supports it.
- Prefer references backed by papers, reports, benchmark sources, or extracted web pages over search snippets alone when possible.
- If a field must rely mostly on company-specific evidence, state that external validation is limited in the text for that field.
- If fewer than 2 solid references are available for a field, explain the limitation in `evidence_gap`.
- Keep each field concise enough to be embedded in a report.
