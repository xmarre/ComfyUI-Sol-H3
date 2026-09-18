# Production arithmetic validation and progressive continuation performance

Status: investigation checkpoint; not an implementation specification yet.

Audited source: Sol 93b3e03f2b7b579aaf55fa0f87f55083b259e25c, Flow 85b953c2cdf8304dbb7da283a9131a3722ff5386, VDN 1bc9f9cd0f685a28f84664b50951df8241501249.

Primary evidence: metrics_00504_.json and Pasted text(20260917-214202).txt; metrics_00511_.json and Pasted text(20260918-020148).txt.

Verified observations:

- Sampler totals: 278.306609 s versus 366.975348 s (+88.668739 s).
- Continuation: 173.924451 s versus 224.178864 s (+50.254414 s).
- Whole schedules: 17L/13A/4F versus 18L/14A/4F; the difference is the intended progressive probe.
- Ordinary gate totals: 0.285472 s versus 42.794905 s. There are 15 distinct ordinary geometry keys in each run; none repeat across lifetimes within the run.
- The 11 partitioned gates in 00511 have no timing field. Their time is unknown, not zero.
- sparse.attention runs an additional all-selected Sol call, dense SDPA and error_metrics, then a separate production sparse call. It does not validate sparse approximation quality.
- gate_wall_s includes any first-use CuTe compilation through interface._compile_sm120, preprocessing, allocations, launch and synchronizing reductions. kernel_loader_s excludes that compilation.
- The process-local compiled cache already survives Request teardown; successful arithmetic keys do not.
- Ordinary 00511 gate costs near 2.8 s for distinct shapes suggest cold compilation. This is not established without compiler telemetry.
- Continuation last high actual is 42.236850 s with no new gate, versus roughly 26.4-26.6 s in 00504. Visible gates do not explain the whole regression.
- 00511 rebuilds VDN/LoRA/Sol setup and has Spectrum profile misses; 00504 starts with profile hits. These are not controlled cold/hot counterparts.

No production runtime files have been changed. The final design must separate compiled-executable reuse, arithmetic acceptance and request/provider/history ownership. Removing validation cannot remove compilation required by a genuinely new production specialization.
