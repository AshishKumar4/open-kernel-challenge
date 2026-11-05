"""Import a built submission with torch / triton stubbed: catches load-order and name errors before a judge run.
python3 harness/import_check.py <file>..."""
import importlib.util, sys
from unittest import mock
import importlib.abc, importlib.machinery


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "triton"):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock()
        m.__path__, m.__spec__ = [], spec
        if spec.name == "torch":  # no GPU here: import-time device probes take the no-CUDA branch
            m.cuda.is_available.return_value = False
            m.cuda.get_device_capability.return_value = (9, 0)
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Stub())
bad = 0
for f in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location("okc_import_check", f)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
        print("ok  ", f)
    except Exception as e:
        bad += 1
        print("FAIL", f, type(e).__name__, str(e)[:200])
sys.exit(1 if bad else 0)
