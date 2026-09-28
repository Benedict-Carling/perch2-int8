# Private destinations

GitHub: https://github.com/Benedict-Carling/perch2-int8

Hugging Face: https://huggingface.co/BenedictCarling/perch2-int8

Both destinations were created and verified **private** on 28 September 2026.
The first Hub upload contains the inference files and model card; research code
and test reports are linked to GitHub. Uploaded through the signed-in browser,
without creating or saving an API token.

- Hub commit: `fe35768b0a0e7c048048ca304349a257325f5d4c`.
- Source Git commit: `d39ff13` (full identifier is in the Hub's `SOURCE_COMMIT`).
- The Hub's displayed INT8 SHA256 matches `SHA256SUMS`:
  `0d3861c911723462306ec1a82f832c17bca3fd58de5bcd937014affaefda7e78`.

For future scripted updates, use an authenticated account. Do not put tokens
in this repository, commit messages, command arguments or chat.

Sign in locally using an existing token authorised to create/write the private
model repository:

```bash
uvx --from huggingface_hub hf auth login
```

Then, from the repository root:

```bash
uv run --no-project --python 3.12 --with huggingface_hub python tools/upload_private_hf.py
```

The default destination is `<signed-in-user>/perch2-int8`. Supply `--repo-id`
only when intentionally choosing a different account/organisation namespace.
The uploader creates the model repository with `private=True`, refuses an
existing public destination, checks privacy again afterwards, and uploads only
committed Git-tracked files. No token is saved into the release tree.

After future uploads, verify the private page and file hashes,
and record the Hub commit alongside the GitHub source commit. Review both pages
while private. Public release is a separate decision requiring Benedict's
explicit instruction.
