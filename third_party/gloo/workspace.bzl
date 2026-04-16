"""Provides the repository macro to import Gloo."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Gloo."""

    GLOO_COMMIT = "80a24ec9bb87af16106514401ed5ef5b1875116b"
    GLOO_SHA256 = "e49b2fc823dc8c82fb37074dade99ac78106e6cb9e0dd73cf85dc04970fbc458"

    tf_http_archive(
        name = "gloo",
        sha256 = GLOO_SHA256,
        strip_prefix = "gloo-{commit}".format(commit = GLOO_COMMIT),
        urls = tf_mirror_urls("https://github.com/qnx-ports/gloo/archive/{commit}.tar.gz".format(commit = GLOO_COMMIT)),
        build_file = "//third_party/gloo:gloo.BUILD",
    )
