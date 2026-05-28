# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a Vulnerability

Report security vulnerabilities to: security@cryptosentinel.ai

Do NOT open public GitHub issues for security vulnerabilities.

## Security Measures

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
- Tabular models use zero-padding for live features — documented limitation
