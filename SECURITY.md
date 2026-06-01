# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a Vulnerability

Report security vulnerabilities to: security@cryptosentinel.ai

Do NOT open public GitHub issues for security vulnerabilities.

## Security Measures

- **Secret scanning:** gitleaks in pre-commit + CI (full-history scan); verified zero leaks across all commits
- **Auth:** RS256 JWT enforced on all protected routes (RBAC), refresh flow, Redis-backed JTI revocation (logout/compromise)
- **Dependency scanning:** pip-audit on every commit
- **Container scanning:** Trivy CRITICAL/HIGH severity scan
- **SBOM:** CycloneDX + SPDX generated on every release
- **JWT:** RS256 asymmetric signing, 15-minute token lifetime
- **PQC:** ML-DSA-65 + ML-KEM-768 (NIST FIPS 203/204)
- **Secrets:** HashiCorp Vault in production (never in code)
- **Non-root containers:** UID 1001 in Docker
- **Network policies:** Zero-trust Kubernetes NetworkPolicy

## Known Accepted Risks

- liboqs compiles at container startup (~2min) — tracked issue
- Elliptic-trained models don't transfer to live features (zero-padding) — RESOLVED by a live-native model trained on the real 16-feature space (see MODELS.md); Elliptic models now power the offline benchmark only
- Live-native model uses weak labels (OFAC + community darklists) — documented in BENCHMARKS.md
