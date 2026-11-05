"""Run identities: what program a measurement measured.

    python3 harness/identity.py <candidate.py>...     # prints file md5, device-source md5 per candidate

file md5   : the built candidate file, byte for byte.
device md5 : the module's final source-string state after import (torch stubbed, as in freeze.py): every module-level
             string (and strings inside module-level lists / tuples / dicts) of 200+ characters, i.e. the CUDA / C++
             sources the candidate compiles. freeze.py preserves this state, so a frozen file and its build share it;
             two arms with different file md5 and the same device md5 differ only in host code. Patch-time temporaries
             (freeze.py's TEMPS) are excluded."""
import hashlib, importlib.abc, importlib.machinery, importlib.util, json, pathlib, sys
from unittest import mock


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "triton"):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock()
        m.__path__, m.__spec__ = [], spec
        return m

    def exec_module(self, module):
        pass


# patch-time temporaries freeze.py drops (harness/freeze.py TEMPS): not part of what runs
TEMPS = {"_o", "_k", "_v", "_n", "_c", "_s", "_a", "_b", "_DR_OLD", "_DR_NEW", "_OKC_ORIG_0"}


def file_md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def _strings(v, out, key):
    if isinstance(v, str):
        if len(v) >= 200:
            out[key] = v
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            _strings(x, out, f"{key}[{i}]")
    elif isinstance(v, dict):
        for k, x in v.items():
            if isinstance(k, (str, int, tuple)):
                _strings(x, out, f"{key}[{k!r}]")


def device_md5(path):
    """md5 of the final module-level source strings; None when the stubbed import fails."""
    if not any(isinstance(f, _Stub) for f in sys.meta_path):
        sys.meta_path.insert(0, _Stub())
    name = f"_ident_{file_md5(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        return None
    finally:
        sys.modules.pop(name, None)
    out = {}
    for k, v in vars(m).items():
        if not k.startswith("__") and k not in TEMPS:
            _strings(v, out, k)
    return hashlib.md5(json.dumps(sorted(out.items())).encode()).hexdigest()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(file_md5(p), device_md5(p), p)
