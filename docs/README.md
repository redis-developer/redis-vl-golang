# RedisVL for Golang documentation

This folder is an [Antora](https://antora.org) project that builds the
documentation site published at
https://redis-developer.github.io/redis-vl-golang/ — the same stack and
branding as the [redis-vl-java docs](https://redis.github.io/redis-vl-java/).

## Layout

```
docs/
├── antora-playbook.yml      # site assembly: sources, UI bundle, attributes
├── package.json             # Antora CLI + site generator (npm)
├── content/
│   ├── antora.yml           # component descriptor: name, version, attributes
│   └── modules/ROOT/
│       ├── nav.adoc         # sidebar navigation
│       └── pages/*.adoc     # the guide pages (AsciiDoc)
└── supplemental-ui/         # Redis branding over the Spring Antora UI bundle
```

## Building locally

Requires Node.js 20+:

```bash
make docs-deps     # npm install (once)
make docs-build    # renders to docs/build/site
make docs-serve    # build + serve at http://localhost:5000
```

## Publishing

`.github/workflows/docs.yml` builds and deploys the site to GitHub Pages on
every push to `main` that touches `docs/` (and on manual dispatch). The
workflow injects the release version from `version.go` into the component
descriptor, so `display_version` here does not need manual bumping.

One-time repository setup: **Settings → Pages → Source: GitHub Actions.**

## Writing pages

Pages are AsciiDoc. Add new pages under `content/modules/ROOT/pages/` and
list them in `nav.adoc`. Useful attributes (defined in `content/antora.yml`):
`{redisvl-version}`, `{url-redisvl-golang}`, `{url-pkg-go-dev}`,
`{url-redis}`, `{url-langcache}`. Cross-link pages with
`xref:page-name.adoc[Link Text]`.

Every Go snippet in the docs must compile against the current API — treat
code in docs with the same review rigor as code in the library.
