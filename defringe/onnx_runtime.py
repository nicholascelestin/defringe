PROVIDER_LABEL = {"CUDAExecutionProvider": "CUDA", "CoreMLExecutionProvider": "MPS",
                  "DmlExecutionProvider": "DirectML", "CPUExecutionProvider": "CPU"}
PROVIDER_ORDER = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider",
                  "CPUExecutionProvider"]


def best_device_name():
    providers = _available_providers()
    if not providers:
        return "CPU (onnxruntime not loaded)"
    return PROVIDER_LABEL.get(providers[0], providers[0])


def make_session(onnx_path):
    import onnxruntime as ort
    error = None
    for provider in _available_providers():
        try:
            with_fallback = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
            session = ort.InferenceSession(str(onnx_path), providers=with_fallback)
            chosen = session.get_providers()[0]
            return session, PROVIDER_LABEL.get(chosen, chosen)
        except Exception as e:
            error = e
    raise error or RuntimeError("no execution provider available")


def _available_providers():
    try:
        import onnxruntime as ort
        installed = set(ort.get_available_providers())
    except Exception:
        return []
    ordered = [p for p in PROVIDER_ORDER if p in installed]
    if "CPUExecutionProvider" not in ordered:
        ordered.append("CPUExecutionProvider")
    return ordered
