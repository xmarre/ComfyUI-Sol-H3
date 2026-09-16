"""Small duck-typed contracts shared with optional attention/forecast providers."""

from dataclasses import dataclass

HISTORY_KEY = "attention_backend_history_v1"
RECEIPTS_KEY = "attention_backend_receipts_v1"
VDN_KEY = "vdn_softmax_provider_v1"
VDN_KEY_V2 = "vdn_softmax_provider_v2"
VDN_KEY_V3 = "vdn_softmax_provider_v3"
VDN_KEY_V4 = "vdn_softmax_provider_v4"
VDN_PREPROCESS_KEY = "vdn_attention_preprocess_v1"
SPECTRUM_EXTERNAL_RUNTIME_KEY = "spectrum_h3_external_patch_runtime"
FLOW_STAGE_KEY = "h3_flow_stage"
FLOW_REFINEMENT_KEY = "h3_refinement"
ATTENTION_MEASURE_KEY = "attention_measure_v1"


def provider_name(provider):
    if provider is None:
        return "comfy.default"
    return (
        f"{getattr(provider, '__module__', '<unknown>')}.{getattr(provider, '__qualname__', type(provider).__name__)}"
    )


def receipt(options, block, route, *, measure_plan=None, call_token=None, fields=None):
    if fields is not None and route == "vdn_local_sol_mapped_v1":
        # This is called only after the mapped kernel returns successfully. Keep a
        # small immutable proof on the request even if its device descriptor is
        # later evicted from the bounded LRU; receipt acceptance must not rely on
        # the current cache contents.
        from .runtime import _REQUEST

        state = _REQUEST.get()
        if state is not None:
            owned = getattr(state, "mapped_validated_receipts", None)
            if owned is None:
                owned = set()
                state.mapped_validated_receipts = owned
            owned.add((block, fields))

    sink = options.get(RECEIPTS_KEY)
    if sink is None:
        return
    if fields is not None:
        if measure_plan is not None:
            raise RuntimeError("attention receipt cannot combine explicit and weighted fields")
        sink.append(("sol_h3", block, route, fields))
        return
    if measure_plan is None:
        sink.append(("sol_h3", block, route))
        return
    from .weighted_measure import receipt_fields

    sink.append(("sol_h3", block, route, receipt_fields(measure_plan, call_token=call_token)))


def _flow_progressive_high_continuation(options):
    """Recognize Flow's explicit later-stage continuation contract."""
    if options.get(FLOW_STAGE_KEY) != "high":
        return False
    refinement = options.get(FLOW_REFINEMENT_KEY)
    return bool(
        isinstance(refinement, dict)
        and refinement.get("api") == 1
        and refinement.get("active") is True
        and refinement.get("source") == "h3_flow_progressive_handoff"
        and refinement.get("min_actual_prefix_steps") == 1
        and refinement.get("sigma_reference") == 1.0
    )


def dense_evaluation_warmup(config, evaluation, options):
    continuation_consumed_default = bool(
        config.dense_evaluations == 1 and _flow_progressive_high_continuation(options)
    )
    return bool(evaluation < config.dense_evaluations and not continuation_consumed_default)


def _freeze_history_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_history_value(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_history_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_history_value(item) for item in value), key=repr))
    return repr(value)


def _closure_values(function):
    code = getattr(function, "__code__", None)
    cells = getattr(function, "__closure__", None)
    if code is None or cells is None or len(code.co_freevars) != len(cells):
        return None
    values = {}
    for name, cell in zip(code.co_freevars, cells):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            return None
    return values


def _mapped_vdn_history_identity(forward, options, layout):
    """Describe provider-v4 geometry without allocating CUDA metadata."""
    describe = getattr(forward, "vdn_query_position_plan_v1", None)
    if not callable(describe):
        return False, None
    summary = describe(options, layout)
    if summary is None:
        return True, None
    if (
        getattr(summary, "tag", None) != "vdn_query_position_plan_v1"
        or type(getattr(summary, "schema", None)) is not int
        or getattr(summary, "schema", None) != 1
        or getattr(summary, "mode", None) not in {"native", "grouped"}
        or not isinstance(getattr(summary, "owner_generation", None), str)
        or not getattr(summary, "owner_generation", "")
        or not isinstance(getattr(summary, "plan_digest", None), str)
        or not isinstance(getattr(summary, "groups", None), tuple)
    ):
        return True, None

    common_identity = (
        summary.owner_generation,
        summary.plan_digest,
        getattr(summary, "seq_len", None),
        getattr(summary, "video_start", None),
        getattr(summary, "video_end", None),
        getattr(summary, "num_frames", None),
        getattr(summary, "tokens_per_frame", None),
        getattr(summary, "anchor_frames", None),
    )
    if summary.mode == "native":
        if summary.groups:
            return True, None
        return True, ("vdn_h3_native_query_position_plan_v1", *common_identity)

    from .mapped_neighbors import MappingUnavailable, POLICY, compile_descriptor, validate_wire_map

    group_identities = []
    for wire in summary.groups:
        try:
            if not isinstance(wire, tuple) or len(wire) != 9:
                return True, None
            validated = validate_wire_map(
                wire, q_rows=wire[5], kv_rows=wire[6], sink_rows=wire[7]
            )
            descriptor = compile_descriptor(validated)
        except MappingUnavailable:
            # A bounded mapping fallback is actual-only in history-v1.
            return True, None
        if validated.owner_generation != summary.owner_generation or validated.plan_digest != summary.plan_digest:
            return True, None
        route_class = "ordinal_existing" if descriptor is None else "mapped_neighbor_additive"
        group_identities.append(
            (
                validated.group_index,
                validated.q_rows,
                validated.kv_rows,
                validated.sink_rows,
                validated.map_digest,
                None if descriptor is None else descriptor.descriptor_digest,
                route_class,
            )
        )
    return True, (
        "vdn_h3_grouped_query_positions_v1",
        *common_identity,
        POLICY,
        tuple(group_identities),
    )


def _vdn_history_identity(forward, options, layout):
    """Describe known VDN routing, preferring the explicit provider-v4 plan hook."""
    has_mapped_hook, mapped = _mapped_vdn_history_identity(forward, options, layout)
    if has_mapped_hook:
        return mapped

    describe = getattr(forward, "attention_history_v1", None)
    if callable(describe):
        return describe(options, layout)

    values = _closure_values(forward)
    if values is None:
        return None
    state = values.get("state")
    cfg = values.get("cfg", getattr(state, "cfg", None))
    base_branch = values.get("base_branch")
    backend = getattr(state, "softmax_backend", None)
    if backend != "grouped" or not isinstance(cfg, dict):
        return None
    if state is None or getattr(state, "cfg", cfg) is not cfg:
        return None

    branch_identity = (
        "none" if base_branch is None else provider_name(type(base_branch)),
        bool(getattr(base_branch, "enable_text_state", False)) if base_branch is not None else False,
    )
    return (
        "vdn_h3_grouped_closure_v1",
        id(forward),
        _freeze_history_value(cfg),
        branch_identity,
    )


def _diffaid_runtime_identity(options):
    raw = options.get(SPECTRUM_EXTERNAL_RUNTIME_KEY)
    if raw is None:
        return ()
    entries = raw if isinstance(raw, (tuple, list)) else (raw,)
    identities = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("provider") != "comfyui-diffaid-patches" or entry.get("schema_version") != 1:
            continue
        instance = entry.get("instance_id")
        if not isinstance(instance, str) or not instance:
            continue
        identities.append(("comfyui-diffaid-patches", 1, instance))
    return tuple(sorted(set(identities)))


def _diffaid_replacement_identity(patch, options):
    if type(patch).__name__ != "MiniMaxH3BlockReplacePatch":
        return None
    runtime_identity = _diffaid_runtime_identity(options)
    if not runtime_identity or not hasattr(patch, "existing_patch"):
        return None
    config = getattr(patch, "config", None)
    fields = (
        "strength",
        "sigma_start",
        "sigma_end",
        "sigma_ramp",
        "token_weight_mode",
        "token_tail",
        "cond_only",
    )
    if config is None or any(not hasattr(config, name) for name in fields):
        return None
    config_identity = tuple((name, _freeze_history_value(getattr(config, name))) for name in fields)
    return (
        ("spectrum_declared_diffaid_h3_v1", runtime_identity, config_identity),
        patch.existing_patch,
    )


def _flow_layout_replacement_identity(patch, block_index):
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (module == "h3_flow_regenerate.attention" or module.endswith(".h3_flow_regenerate.attention"))
        and qualname.endswith("make_layout_block_wrapper.<locals>.wrapper")
        and getattr(patch, "_h3_flow_layout_wrapper", False) is True
    ):
        return None

    scope = getattr(patch, "_h3_flow_layout_scope", None)
    if scope not in {"layout", "attention"}:
        return None
    marker_previous = getattr(patch, "_h3_flow_previous", None)
    marker_metrics = getattr(patch, "_h3_flow_metrics", None)
    values = _closure_values(patch)
    required = {"layer", "metrics", "previous", "record_layout"}
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None
    if values["previous"] is not marker_previous or values["metrics"] is not marker_metrics:
        return None
    if type(values["record_layout"]) is not bool:
        return None

    record_layout = values["record_layout"]
    if scope == "layout":
        if block_index != 0 or record_layout is not True:
            return None
    elif record_layout != (block_index == 0):
        return None

    return (
        ("h3_flow_layout_wrapper_v1", scope, block_index, record_layout),
        marker_previous,
    )


def _flow_mixed_grid_replacement_identity(patch, block_index):
    module = str(getattr(patch, "__module__", ""))
    qualname = str(getattr(patch, "__qualname__", ""))
    if not (
        (module == "h3_flow_regenerate.mixed_grid" or module.endswith(".h3_flow_regenerate.mixed_grid"))
        and qualname.endswith("mixed_diffusion_wrapper.<locals>.wrap.<locals>.call")
    ):
        return None

    values = _closure_values(patch)
    required = {"layer", "previous", "plan", "layout", "mixed_layout", "va", "vb", "old_prefix", "inner"}
    if values is None or not required.issubset(values):
        return None
    if type(values["layer"]) is not int or values["layer"] != int(block_index):
        return None

    plan = values["plan"]
    layout = values["layout"]
    mixed_layout = values["mixed_layout"]
    inner = values["inner"]
    try:
        prefix_t = int(plan.prefix_t)
        temporal = int(plan.temporal)
        source_rows = int(plan.source_rows)
        target_rows = int(plan.target_rows)
        prefix_rows = int(plan.prefix_rows)
        mixed_rows = int(plan.mixed_rows)
        target_hw = tuple(int(value) for value in plan.target_hw)
        va = int(values["va"])
        vb = int(values["vb"])
        old_prefix = int(values["old_prefix"])
        native_rows = int(layout.seq_len)
        mixed_sequence_rows = int(mixed_layout.seq_len)
        native_segments = tuple(layout.segments)
        mixed_segments = tuple(mixed_layout.segments)
        inner_blocks = len(inner.blocks)
        attention_measure = plan.attention_measure
    except (AttributeError, TypeError, ValueError):
        return None

    if (
        not 0 < prefix_t < temporal
        or source_rows <= 0
        or target_rows <= source_rows
        or type(attention_measure) is not bool
        or len(target_hw) != 2
        or any(value <= 0 or value % 2 for value in target_hw)
        or prefix_rows != prefix_t * target_rows
        or old_prefix != prefix_t * source_rows
        or mixed_rows != prefix_rows + (temporal - prefix_t) * source_rows
        or va < 0
        or vb != va + temporal * source_rows
        or native_rows != vb
        or mixed_sequence_rows != va + mixed_rows
        or not native_segments
        or native_segments[-1] != (va, vb, "video")
        or not mixed_segments
        or mixed_segments[-1] != (va, mixed_sequence_rows, "video")
        or not isinstance(getattr(mixed_layout, "signature", None), tuple)
        or not mixed_layout.signature
        or mixed_layout.signature[0] != "h3_flow_mixed_grid_v1"
        or block_index < 0
        or block_index >= inner_blocks
    ):
        return None

    identity = (
        "h3_flow_mixed_grid_v1",
        native_rows,
        mixed_sequence_rows,
        va,
        temporal,
        prefix_t,
        source_rows,
        target_rows,
        target_hw,
        attention_measure,
        repr(getattr(layout, "signature", None)),
        repr(mixed_layout.signature),
    )
    return identity, values["previous"]


def _replacement_history_identity(patch, block_index, config, options):
    from .runtime import BlockPatch

    current = patch
    seen = set()
    chain = []
    sol_count = 0
    while current is not None:
        if id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, BlockPatch):
            if current.index != block_index or current.config != config:
                return None
            sol_count += 1
            if sol_count != 1:
                return None
            chain.append(("sol_h3_block_patch", config.metadata()["fingerprint"]))
            current = current.previous
            continue
        declared = _flow_layout_replacement_identity(current, block_index)
        if declared is None:
            declared = _flow_mixed_grid_replacement_identity(current, block_index)
        if declared is None:
            declared = _diffaid_replacement_identity(current, options)
        if declared is None:
            return None
        identity, current = declared
        chain.append(identity)
    if sol_count != 1:
        return None
    return tuple(chain)


@dataclass(frozen=True)
class HistoryPolicy:
    config: object

    def __call__(self, *, layout, options, model):
        from .runtime import _REQUEST

        state = _REQUEST.get()
        if state is None:
            return None
        if options.get(ATTENTION_MEASURE_KEY) is not None:
            return None
        replacements = options.get("patches_replace", {}).get("dit", {})
        replacement_identity = []
        for index in range(len(model.blocks)):
            patch = replacements.get(("double_block", index))
            identity = _replacement_history_identity(patch, index, self.config, options)
            if identity is None:
                return None
            replacement_identity.append(identity)
        vdn = []
        for block in model.blocks:
            forward = block.attn.forward
            if getattr(forward, "_vdn_forward", False):
                identity = _vdn_history_identity(forward, options, layout)
                if identity is None:
                    return None
                vdn.append(identity)
        signature = getattr(layout, "signature", None)
        phase = "dense" if dense_evaluation_warmup(self.config, state.evaluations, options) else "sol"
        return (
            self.config.metadata()["fingerprint"],
            phase,
            repr(signature),
            getattr(layout, "seq_len", None),
            tuple(getattr(layout, "segments", ())),
            str(getattr(model, "dtype", None)),
            tuple(replacement_identity),
            provider_identity(
                options.get("optimized_attention_override"),
                state.disabled_dense_providers,
            ),
            tuple(vdn),
        )

    def accept_receipts(self, receipts):
        if not receipts:
            return False
        plain_routes = {
            "sol",
            "sol_external_mixed",
            "sol_external_mixed_measure",
            "dense_warmup",
            "vdn_local_sol",
            "vdn_dense_warmup",
            "vdn_local_native",
            "vdn_global_native",
            "vdn_anchor_native",
            "vdn_flex_masked_native",
            "external_sequence_native",
        }
        from . import weighted_measure
        from .mapped_neighbors import POLICY, RECEIPT_TAG
        from .runtime import _REQUEST

        state = _REQUEST.get()
        owned_mapped = getattr(state, "mapped_validated_receipts", set()) if state is not None else set()

        for item in receipts:
            if not isinstance(item, tuple) or len(item) not in {3, 4} or item[0] != "sol_h3":
                return False
            if type(item[1]) is not int or item[1] < 0:
                return False
            route = item[2]
            if not isinstance(route, str):
                return False
            if len(item) == 3:
                # Mapping fallbacks are deliberately actual-only in history v1.
                if route.startswith("vdn_local_native_mapping:"):
                    return False
                if route not in plain_routes:
                    return False
                continue

            fields = item[3]
            if route == "vdn_local_sol_mapped_v1":
                if not isinstance(fields, tuple) or len(fields) != 12:
                    return False
                (
                    key,
                    owner_generation,
                    plan_digest,
                    group_index,
                    q_rows,
                    kv_rows,
                    sink_rows,
                    map_digest,
                    descriptor_digest,
                    policy,
                    kernel_contract,
                    completed,
                ) = fields
                if (
                    key != RECEIPT_TAG
                    or not isinstance(owner_generation, str) or not owner_generation
                    or not isinstance(plan_digest, str) or len(plan_digest) != 64
                    or type(group_index) is not int or group_index < 0
                    or type(q_rows) is not int or q_rows <= 0
                    or type(kv_rows) is not int or kv_rows <= 0
                    or type(sink_rows) is not int or not 0 <= sink_rows <= kv_rows
                    or not isinstance(map_digest, str) or len(map_digest) != 64
                    or not isinstance(descriptor_digest, str) or len(descriptor_digest) != 64
                    or policy != POLICY
                    or not isinstance(kernel_contract, str) or not kernel_contract
                    or completed is not True
                    or (item[1], fields) not in owned_mapped
                ):
                    return False
                continue

            if route == "sol_external_mixed_weighted_measure":
                expected_profile = weighted_measure.SPARSE_IMPLEMENTATION_PROFILE
                expected_route = weighted_measure.SPARSE_NUMERICAL_ROUTE
            elif route == "dense_warmup" or route.startswith("kernel_unavailable:"):
                expected_profile = weighted_measure.DENSE_IMPLEMENTATION_PROFILE
                expected_route = weighted_measure.DENSE_NUMERICAL_ROUTE
            else:
                return False

            if not isinstance(fields, tuple) or len(fields) != 11:
                return False
            (
                key,
                call_token,
                owner_generation,
                semantic_digest,
                implementation_profile,
                numerical_route,
                q_rows,
                kv_rows,
                exact_range_digest,
                preprocess_digest,
                completed,
            ) = fields
            if (
                key != ATTENTION_MEASURE_KEY
                or not isinstance(call_token, tuple)
                or len(call_token) != 2
                or call_token[0] != "sol_h3_evaluation"
                or type(call_token[1]) is not int
                or call_token[1] < 0
                or not isinstance(owner_generation, str)
                or not owner_generation
                or not isinstance(semantic_digest, str)
                or not semantic_digest
                or implementation_profile != expected_profile
                or numerical_route != expected_route
                or type(q_rows) is not int
                or q_rows <= 0
                or type(kv_rows) is not int
                or kv_rows <= 0
                or not isinstance(exact_range_digest, str)
                or not exact_range_digest
                or not isinstance(preprocess_digest, str)
                or not preprocess_digest
                or completed is not True
            ):
                return False
        return True


def provider_identity(provider, disabled_dense_providers=()):
    transforms = []
    seen = set()
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            return ("cyclic_preprocess", id(provider))
        seen.add(id(provider))
        transform, provider = provider.attention_preprocess_v1
        transforms.append(provider_name(transform))
    identity = (tuple(transforms), provider_name(provider), id(provider))
    if id(provider) in disabled_dense_providers:
        return (*identity, "unavailable_fallback")
    return identity
