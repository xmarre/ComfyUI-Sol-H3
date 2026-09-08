from .contracts import Config


class SolH3Exact:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimizations/Sol-H3"
    DESCRIPTION = "Fuse native H3 affine modulation with intermediate rounding preserved. CUDA required; GPU validation pending."

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
                "dense_layers": ("INT", {"default": 2, "min": 0, "max": 100})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/optimizations/Sol-H3/experimental"
    DESCRIPTION = "Approximate Sol-Attn through ComfyUI's comfy-kitchen kernel on eligible SM120 H3/VDN calls; unsupported calls fall back locally."

    def apply(self, model, exact_fusion=True, tau=1.0, dense_evaluations=1, dense_layers=2):
        from .runtime import install
        return (install(model, Config(exact_fusion, "sol", tau, dense_evaluations, dense_layers)),)


NODE_CLASS_MAPPINGS = {"SolH3Exact": SolH3Exact, "SolH3Experimental": SolH3Experimental}
NODE_DISPLAY_NAME_MAPPINGS = {"SolH3Exact": "Sol-H3 Exact Runtime",
                              "SolH3Experimental": "Sol-H3 SOL Attention (Experimental)"}
