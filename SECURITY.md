# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

The development team takes security seriously, especially regarding credential management, external API authentication, and subprocess execution security.

### 1. Scope of Vulnerabilities
- Insecure storage or leakage of API tokens (e.g., Lichess Bot API tokens).
- Path traversal or arbitrary code execution vulnerabilities during UCI engine subprocess dispatch.
- Insecure deserialization of model weights or checkpoints.

### 2. How to Report
**Please DO NOT open a public GitHub issue for security vulnerabilities.**

Instead, report vulnerabilities privately by emailing the maintainers at:
`security-chess-vision-ai@proton.me` (or via private GitHub Security Advisories).

Please include:
- A clear description of the vulnerability.
- Steps to reproduce or proof-of-concept script.
- Potential impact and suggested mitigations.

### 3. Response Process
- You will receive an acknowledgment within **48 hours**.
- A patch will be prioritized and tested.
- A CVE / security advisory will be coordinated and released alongside the fix.
