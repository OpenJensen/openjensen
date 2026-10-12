# Security and deployment scope

OPEN JENSEN is a single-user, self-hosted research application. Keep the API on its default loopback address. Use an SSH tunnel when accessing a remote installation. The core API is not an authenticated multi-user Internet service; publishing it directly is outside the documented deployment scope.

Keep workspace directories, backups, provider credentials, tokens and private runtime configurations accessible only to their owner. Configure worker interpreters and model loaders from trusted operator-controlled paths. Review cloud billing consent and inspect job cleanup before assuming a paid resource has stopped.

## Reporting a security concern

Check the [repository Security tab](https://github.com/OpenJensen/openjensen/security) for an available private reporting option. If one is available, use it for vulnerability details. Otherwise, open a [GitHub issue](https://github.com/OpenJensen/openjensen/issues) requesting a private reporting route without including exploit details, secrets or private user data.

Include the affected source revision/version, installation mode and enough redacted information to reproduce the concern. No response-time or security certification guarantee is claimed.
