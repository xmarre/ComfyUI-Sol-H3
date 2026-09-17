from types import SimpleNamespace

from sol_h3 import interop, weighted_measure


def _plan(profile, route):
    return SimpleNamespace(
        owner_generation="owner-generation",
        semantic_digest="semantic-digest",
        implementation_profile=profile,
        numerical_route=route,
        q_rows=44958,
        kv_rows=44958,
        exact_range_digest="exact-range-digest",
        preprocess_digest="preprocess-digest",
    )


def _weighted_receipt(route, profile, numerical_route):
    plan = _plan(profile, numerical_route)
    return (
        "sol_h3",
        0,
        route,
        weighted_measure.receipt_fields(plan, call_token=1),
    )


def test_history_policy_accepts_completed_weighted_dense_and_sparse_receipts():
    policy = interop.HistoryPolicy(SimpleNamespace())
    dense = _weighted_receipt(
        "dense_warmup",
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    sparse = _weighted_receipt(
        "sol_external_mixed_weighted_measure",
        weighted_measure.SPARSE_IMPLEMENTATION_PROFILE,
        weighted_measure.SPARSE_NUMERICAL_ROUTE,
    )
    fallback = _weighted_receipt(
        "kernel_unavailable:test",
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    assert policy.accept_receipts((dense, sparse, fallback))


def test_history_policy_rejects_weighted_route_profile_mismatch_and_malformed_token():
    policy = interop.HistoryPolicy(SimpleNamespace())
    wrong_route = _weighted_receipt(
        "sol_external_mixed_weighted_measure",
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    assert not policy.accept_receipts((wrong_route,))

    good = _weighted_receipt(
        "dense_warmup",
        weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
        weighted_measure.DENSE_NUMERICAL_ROUTE,
    )
    fields = list(good[3])
    fields[1] = ("sol_h3_evaluation", True)
    malformed = (*good[:3], tuple(fields))
    assert not policy.accept_receipts((malformed,))


def test_history_policy_keeps_legacy_three_field_receipts():
    policy = interop.HistoryPolicy(SimpleNamespace())
    assert policy.accept_receipts((("sol_h3", 0, "dense_warmup"),))
    assert not policy.accept_receipts((("sol_h3", 0, "unknown"),))
