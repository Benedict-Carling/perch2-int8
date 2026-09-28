# Private destinations

GitHub: https://github.com/Benedict-Carling/perch2-int8

Hugging Face upload is prepared but requires an authenticated account. Do not
put tokens in this repository, commit messages, command arguments or chat.

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

After upload, verify the private page and file hashes, add its URL to README,
and record the Hub commit alongside the GitHub source commit. Review both pages
while private. Public release is a separate decision requiring Benedict's
explicit instruction.
