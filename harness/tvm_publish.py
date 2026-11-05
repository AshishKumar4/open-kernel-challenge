"""tvm-ffi modules without shared writable build directories: a module is built in a directory private to the
process that builds it, then published into the cache by one atomic rename; a published directory is never written
again, and a process that finds one only loads its .so (no ninja run, so nothing in it changes). Two writers of one
module race only on the rename: the first publishes, the second loads the published one and drops its own. Shipped
modules (harness/h200/box_ship.sh) are published the same way. Every directory the cache holds is therefore complete
and immutable; .private, .trash and .incoming hold the in-flight and the set-aside ones.

install() patches tvm_ffi.cpp.extension.build_inline, which load_inline calls (okc_harness and cpu_prebuild do this)."""
import contextlib
import os
import pathlib
import shutil
import socket
import time


def _cache():
    return pathlib.Path(os.environ.get("TVM_FFI_CACHE_DIR", pathlib.Path("~/.cache/tvm-ffi").expanduser())).expanduser()


def published_dir(X, name, cpp_sources=None, cuda_sources=None, functions=None, extra_cflags=None,
                  extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, embed_cubin=None, **_):
    """the cache directory tvm-ffi's build_inline would build these sources in (its own naming, X its extension module)"""
    cpp_list, cuda_list = X._str_seq2list(cpp_sources), X._str_seq2list(cuda_sources)
    cpp, cuda = "\n".join(cpp_list), "\n".join(cuda_list)
    if functions is None:
        fmap = {}
    elif isinstance(functions, str):
        fmap = {functions: ""}
    elif isinstance(functions, dict):
        fmap = dict(functions)
    else:
        fmap = {f: "" for f in functions}
    if cpp_list:
        cpp, cuda = X._decorate_with_tvm_ffi(cpp, fmap), X._decorate_with_tvm_ffi(cuda, {})
    else:
        cpp, cuda = X._decorate_with_tvm_ffi(cpp, {}), X._decorate_with_tvm_ffi(cuda, fmap)
    h = X._hash_sources(cpp, cuda, None, None, fmap, list(extra_cflags or []), list(extra_cuda_cflags or []),
                        list(extra_ldflags or []), list(extra_include_paths or []), embed_cubin)
    return _cache() / f"{name}_{h}"


def publish(private, shared):
    """private's build into shared by one rename (a directory that exists without its .so, an interrupted build of the
    old in-place scheme, is set aside first); True if shared now holds a complete build"""
    for attempt in (0, 1):
        try:
            os.rename(private, shared)   # (fails if shared exists: the first writer's stays)
            return True
        except OSError:
            if any(shared.glob("*.so")):
                return True
            if attempt == 0:   # (incomplete: set aside, never written in place)
                trash = shared.parent / ".trash"
                trash.mkdir(exist_ok=True)
                with contextlib.suppress(OSError):
                    os.rename(shared, trash / f"{shared.name}.{socket.gethostname()}.{os.getpid()}.{time.time_ns()}")
    return any(shared.glob("*.so"))


def install():
    import tvm_ffi.cpp.extension as X
    if getattr(X.build_inline, "_published", False):
        return
    build = X.build_inline

    def build_inline(*a, **k):
        if a or k.get("build_directory"):   # (an explicit directory, or positional use: as tvm-ffi does it)
            return build(*a, **k)
        shared = published_dir(X, **k)
        so = shared / f"{k['name']}.so"
        if so.exists():
            return str(so)
        private = shared.parent / ".private" / f"{shared.name}.{socket.gethostname()}.{os.getpid()}"
        shutil.rmtree(private, ignore_errors=True)
        built = build(**{**k, "build_directory": str(private)})
        if publish(private, shared):
            shutil.rmtree(private, ignore_errors=True)   # (another writer published first: ours is dropped)
            return str(so)
        return built   # (could not publish: load the private build)

    build_inline._published = True
    X.build_inline = build_inline
