from .contracts import Config


class SolH3Exact:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimizations/Sol-H3"
    DESCRIPTION = "Fuse native H3 affine modulation with intermediate rounding preserved on validated Linux/WSL CUDA. Native Windows delegates to untouched native H3."

    def apply(self, model):
        from .runtime import install
        return (install(model, Config()),)


class SolH3Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                "exact_fusion": ("BOOLEAN", {"default": True}),
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "dense_evaluations": ("INT", {"default": 1, "min": 0, "max": 100}),
                "dense_layers": ("INT", {"default": 2, "min": 0, "max": 100})},
                "optional": {"qkv_chunk_size": ("INT", {"default": 4096, "min": 0, "max": 65536, "step": 64,
                    "tooltip": "Bound native H3 QKV projection rows; retain BF16 K/V and recompute Q. 0 uses the full native projection. Foreign attention owners retain their execution path."})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimizations/Sol-H3/experimental"
    DESCRIPTION = "Approximate Sol-Attn through the packaged Sana CuTe kernel on eligible SM120 H3/VDN calls. The default keeps the first full denoiser evaluation dense for conservative startup trajectory stability. dense_evaluations=0 is an aggressive speed opt-in: with Spectrum it can avoid one actual NFE, but it may cause visible startup temporal discontinuities. SOL kernel execution requires Linux/WSL2; unsupported calls fall back locally."

    def apply(self, model, exact_fusion=True, tau=1.0, dense_evaluations=1, dense_layers=2, qkv_chunk_size=4096):
        from .runtime import install
        return (install(model, Config(exact_fusion, "sol", tau, dense_evaluations, dense_layers, qkv_chunk_size)),)




class SolKrea2Experimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            "min_tokens": ("INT", {"default": 4096, "min": 0, "max": 1048576, "step": 256}),
            "dense_evaluations": ("INT", {"default": 1, "min": 0, "max": 100,
                                          "tooltip": "First N native DiT evaluations use dense attention. With Spectrum Krea 2 warmup=2 and bootstrap=false, 0 aligns both initial actual feature anchors on sparse attention and avoids a reset. The first evaluation then becomes approximate; compare image quality."}),
            "dense_layers": ("INT", {"default": 2, "min": 0, "max": 28}),
        }, "optional": {
            "verbose": ("BOOLEAN", {"default": False}),
            "extra_tokens": ("INT", {"default": 256, "min": 0, "max": 256, "step": 64}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimizations/Sol-H3/experimental"
    DESCRIPTION = (
        "Experimental Sol-Attn for native Krea 2: Sana CuTe BF16 or ComfyUI Kitchen native FP16. "
        "Protects Krea 2 text prefix K/V and text query blocks; FP16 extra_tokens defaults to 256. "
        "Keeps image-reference calls dense until noncontiguous exact-KV routing is supported; "
        "other unsupported calls inherit the previous attention backend."
    )

    def apply(self, model, enabled=True, tau=1.0, min_tokens=4096,
              dense_evaluations=1, dense_layers=2, verbose=False, extra_tokens=256):
        if not enabled:
            return (model,)
        # Old workflows stored 'verbose' in widget position 6. The first
        # FP16 overlay accidentally inserted 'extra_tokens' at that position.
        # Append-only schema above restores the previous layout. Also accept
        # workflows saved with that intermediate, swapped seven-widget layout.
        if type(extra_tokens) is bool:
            if type(verbose) is int and verbose in (0, 64, 128, 192, 256):
                extra_tokens, verbose = verbose, extra_tokens
            else:
                extra_tokens = 256
        from .krea2 import Settings, install
        return (install(
            model,
            Settings(tau=tau, min_tokens=min_tokens,
                     dense_evaluations=dense_evaluations, dense_layers=dense_layers,
                     extra_tokens=extra_tokens),
            verbose=verbose,
        ),)

NODE_CLASS_MAPPINGS = {"SolH3Exact": SolH3Exact, "SolH3Experimental": SolH3Experimental,
                       "SolKrea2Experimental": SolKrea2Experimental}
NODE_DISPLAY_NAME_MAPPINGS = {"SolH3Exact": "Sol-H3 Exact Runtime",
                              "SolH3Experimental": "Sol-H3 SOL Attention (Experimental)",
                              "SolKrea2Experimental": "Sol-Krea2 Sparse Attention (Experimental)"}
