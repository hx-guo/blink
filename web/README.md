# TGF catalog website

Static site (Next.js, `output: "export"`) for the real-time TGF pipeline. It only reads the files that
`blink export` writes to `public/data/`; it computes nothing itself. See `docs/tgf-pipeline-design.md`.

```bash
pnpm install
# real data
blink export <catalog dirs> --registry <registry> --data data -o web/public/data [--public-only]
# or demo data (catalog v6 with simulated light curves; the pages show a demo banner)
python3 web/scripts/make_demo_data.py          # from the repository root
pnpm dev                                       # http://localhost:3000
pnpm build                                     # static site in out/; BASE_PATH=/sub/path for a sub-path
```

The pages fetch `data/*.json` and draw with D3 in the browser; the build reads no data. Deploy `out/`
once, then let the pipeline replace only `out/data/` (or wherever `data/` is served) on each update.

Event links use query strings (`/tgf/?name=170623AA`, `/cand/?id=...`, `/instrument/?slug=...`), which
work on any static host. With a rewrite on the server the path form works too, e.g. nginx:

```nginx
location ~ ^/tgf/[^/]+/?$ { try_files /tgf/index.html =404; }
```

`--public-only` builds the public site (instruments listed in the registry's
`public_instruments.txt`); without it, the internal site shows every instrument and marks the
unvalidated ones.
