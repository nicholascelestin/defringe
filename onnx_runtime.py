"""ONNX Runtime device selection — pick the best available execution provider and build a
session on it, falling back gracefully. Infrastructure: no domain logic, no UI."""

# Providers best-first, with the friendly names the device widget shows.
PROVIDER_LABEL = {"CUDAExecutionProvider": "CUDA", "CoreMLExecutionProvider": "MPS",
                  "DmlExecutionProvider": "DirectML", "CPUExecutionProvider": "CPU"}
PROVIDER_ORDER = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider",
                  "CPUExecutionProvider"]


def available_providers():
    """Installed providers in best-first order; CPU always present as the floor.
    Empty list means onnxruntime itself isn't importable."""
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return []
    ordered = [p for p in PROVIDER_ORDER if p in avail]
    if "CPUExecutionProvider" not in ordered:
        ordered.append("CPUExecutionProvider")
    return ordered


def best_device_name():
    """Highest compatible device, without building a session (for the idle widget)."""
    provs = available_providers()
    if not provs:
        return "CPU (onnxruntime not loaded)"
    return PROVIDER_LABEL.get(provs[0], provs[0])


def make_session(onnx_path):
    """Build an InferenceSession on the highest available provider, falling back through the
    list until one actually loads. Returns (session, friendly_device_name). Raises if
    onnxruntime is missing or no provider initialises."""
    import onnxruntime as ort
    err = None
    for prov in available_providers():
        try:
            provs = [prov] if prov == "CPUExecutionProvider" else [prov, "CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=provs)
            used = sess.get_providers()[0]                 # the one actually in front
            return sess, PROVIDER_LABEL.get(used, used)
        except Exception as e:
            err = e                                        # provider present but wouldn't init -> try the next
    raise err or RuntimeError("no execution provider available")
