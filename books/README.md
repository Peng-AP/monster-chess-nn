# Opening books

The default iteration gate uses sampled free play, not these legacy books.
Training starts remain at `data/start_fens/book_v31_seed.jsonl` (unchanged).

The top level retains books referenced by tools/tests, the pinned partition
manifest and its matching book, the v31 source book, and the recent v48/v49 books.
Forty-one retired book/sidecar files moved to `archive/cleanup_20260906/`.
Their contents are unchanged; old benchmark paths describe their original locations.

The exact model/book archive manifest is
`benchmarks/cleanup_models_books_20260906.json`. Restore the entire cleanup from
the repository root with:

```powershell
& benchmarks/cleanup_models_books_20260906.ps1 -Restore -Apply
```

Omit `-Apply` for a validation-only restore check. Restoration refuses to overwrite
existing files. A single archived item can also be moved back to its `source`
path from the manifest. Nothing was permanently deleted.
