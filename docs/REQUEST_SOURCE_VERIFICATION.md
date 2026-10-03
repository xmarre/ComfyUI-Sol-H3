# Packaged source ownership during partitioned sampling

Partitioned attention verifies the complete packaged Sana source tree before
its first kernel invocation in each native Sol sampling request. Source trust
belongs to the `OUTER_SAMPLE` request. Dense, sparse and arithmetic-reference
kernel invocations in that request share the successful source check.

The request stores one source identity and a lock, without retaining Q/K/V,
outputs, CUDA resources or the parsed source manifest. The identity includes
the source name, revision, kernel contract and verifier function. A changed
identity requires another full check. Concurrent first entries share the
request's check; independent or nested requests validate separately. Returning
from a nested request restores the outer owner through the existing context.

A verification attempt clears previous trust before reading the source. Any
failure propagates before kernel execution and leaves the request unverified.
Success is recorded only after the complete file-set and hash validation.
Sampler failure or cancellation ends the native request lifetime; the next
request must validate again. Explicit standalone kernel calls and direct
`verify_source()` calls retain uncached checks.

This ownership assumes packaged code remains fixed during a sampling request,
as with the released request-owned backend loader. Installed source updates
require restarting the application. Files changed between requests are checked
again before the next partitioned kernel. This cache is not a file watcher or
a mechanism for replacing live compiled kernels.

Each sampling summary records:

- `partitioned_source_tree_verified`: the request has a successful current
  source identity, independently of overall sampling success.
- `partitioned_source_verification_calls`: full verification attempts,
  including attempts that fail. An unused partitioned path reports zero.
- `partitioned_source_verification_wall_s`: CPU wall time in those attempts,
  excluding kernel execution and arithmetic-reference validation.

These source receipts do not qualify GPU output or speed. Device, shape,
strides, key measure and numerical ownership retain their independent checks.
Weighted dense and sparse arithmetic gates, completion receipts, attention
operations and numerical forecasting identities are unchanged.
