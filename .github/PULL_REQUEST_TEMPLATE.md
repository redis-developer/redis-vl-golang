# Description

<!-- What does this PR change, and why? Link related issues with "Fixes #NN". -->

## Type of change

<!-- Keep the ones that apply. -->

- Bug fix
- New feature
- Documentation
- Refactoring / maintenance

## Checklist

- [ ] `make check` passes (fmt + vet + tests; Docker running for integration coverage)
- [ ] `make lint` passes
- [ ] If the change touches `extensions/vectorize/hf`: `make work && make test-hf` passes
- [ ] New behavior is covered by tests
- [ ] User-facing changes are reflected in the docs (`docs/content/`) and CHANGELOG.md
- [ ] Python-parity note: if this ports or diverges from redis-vl-python behavior, the PR description says so
