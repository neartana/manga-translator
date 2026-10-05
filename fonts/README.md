# Font packs

> Cloned from GitHub? The binary fonts are not committed — run
> `python scripts/download_fonts.py` once to fetch the starter packs
> (Comic Neue + Noto Sans CJK, both OFL-licensed).

The renderer picks fonts in this order:

1. `--font-path` / the "Font file" field in the UI (explicit override)
2. `fonts/<target-lang>/` — the pack matching the language you translate **into**
3. `fonts/default/` — fallback for every language
4. system fonts (DejaVu, Noto, …)

## Adding your own pack

Drop `.ttf` / `.otf` / `.ttc` files into a folder named after the target
language code:

```
fonts/
  en/                 English      (bundled: Comic Neue, OFL)
  zh-CN/              Simplified Chinese
  ko/                 Korean
  fr/                 French
  default/            fallback     (bundled: Noto Sans CJK, OFL — covers ja/zh/ko)
```

Language codes: `en, ja, zh-CN, zh-TW, ko, es, fr, de, it, pt, ru, ar,
th, vi, id, nl, pl, tr, uk`.

If a pack folder contains several files, the first one that loads is used
(alphabetical order), so prefix the file name, e.g. `01-AnimeAce.ttf`.

## Licenses

- Comic Neue — SIL Open Font License 1.1, © Craig Rozynski
- Noto Sans CJK — SIL Open Font License 1.1, © Google / Adobe

Only add fonts you have the rights to use.
