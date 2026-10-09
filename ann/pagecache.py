"""Linux page-cache tools for cold/warm-cache experiments (extra credit).

  residency(paths) : resident pages per file. Uses the util-linux `fincore` CLI when available,
                     otherwise the same mincore(2) syscall fincore uses (via ctypes; mmap
                     without touching pages, so measuring does not change residency).
  evict(paths)     : per-file fsync + posix_fadvise(POSIX_FADV_DONTNEED) (advisory).
  device_info(dir) : block device / model / rotational flag behind a directory.
"""
import ctypes
import ctypes.util
import mmap
import os
import shutil
import subprocess

PAGE = os.sysconf("SC_PAGE_SIZE")


def _mincore(path):
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    npages = (size + PAGE - 1) // PAGE
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            vec = (ctypes.c_ubyte * npages)()
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident = sum(b & 1 for b in bytes(vec))
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)
    return resident, npages


def residency(paths):
    """{path: {resident_pages, total_pages, resident_bytes, resident_frac, tool}}."""
    out = {}
    have_cli = shutil.which("fincore") is not None
    for p in map(str, paths):
        if have_cli:
            r = subprocess.run(["fincore", "--bytes", "--noheadings", "--raw", "--output", "PAGES,SIZE", p],
                               capture_output=True, text=True, check=True).stdout.split()
            res_pages, size = int(r[0]), int(r[1])
            total = (size + PAGE - 1) // PAGE
            tool = "fincore (util-linux)"
        else:
            res_pages, total = _mincore(p)
            tool = "mincore(2) via ctypes (fincore CLI not installed)"
        out[p] = {"resident_pages": res_pages, "total_pages": total,
                  "resident_bytes": res_pages * PAGE,
                  "resident_frac": res_pages / total if total else 0.0, "tool": tool}
    return out


def evict(paths):
    """Advise the kernel to drop cached pages of each file (per file, no global drop_caches)."""
    for p in map(str, paths):
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def device_info(path):
    """Best-effort description of the physical device holding `path`."""
    info = {"path": str(path)}
    try:
        src = subprocess.run(["findmnt", "-n", "-o", "SOURCE", "--target", str(path)],
                             capture_output=True, text=True).stdout.strip()
        info["mount_source"] = src
        pk = subprocess.run(["lsblk", "-no", "PKNAME", src], capture_output=True, text=True).stdout.strip()
        disk = "/dev/" + pk.splitlines()[0] if pk else src
        info["disk"] = disk
        for col in ("MODEL", "ROTA", "TRAN", "SIZE"):
            info[col.lower()] = subprocess.run(["lsblk", "-dno", col, disk], capture_output=True,
                                               text=True).stdout.strip()
        info["rotational"] = {"1": "HDD (rotational)", "0": "SSD/NVMe (non-rotational)"}.get(
            info["rota"], "unknown")
    except Exception as e:                               # noqa: BLE001
        info["error"] = repr(e)
    return info
