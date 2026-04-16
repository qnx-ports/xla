"""Provides the repository macro to import mpitrampoline."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports mpitrampoline."""

    MPITRAMPOLINE_COMMIT = "ba3786b67762be53bb12a712cdb3366e3e117c21"
    MPITRAMPOLINE_SHA256 = "3b078c33673662c53558c6b6d1457d8b12fa6ae5fb723af91f0c60a53778f764"

    tf_http_archive(
        name = "mpitrampoline",
        sha256 = MPITRAMPOLINE_SHA256,
        strip_prefix = "MPItrampoline-{commit}".format(commit = MPITRAMPOLINE_COMMIT),
        urls = tf_mirror_urls("https://github.com/qnx-ports/mpitrampoline/archive/{commit}.tar.gz".format(commit = MPITRAMPOLINE_COMMIT)),
        patch_file = ["//third_party/mpitrampoline:gen.patch"],
        build_file = "//third_party/mpitrampoline:mpitrampoline.BUILD",
    )
