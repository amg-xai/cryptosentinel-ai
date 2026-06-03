"""K8s manifest hardening checks — assert security-critical settings are
present so they can't be silently removed. (Static checks; full runtime
verification requires a cluster.)"""
from pathlib import Path
import yaml
import pytest

K8S = Path("k8s")
pytestmark = pytest.mark.skipif(not K8S.exists(), reason="no k8s dir")


def _load_all(path):
    return list(yaml.safe_load_all(open(path)))


def _api_deployment():
    for doc in _load_all(K8S / "deployment-api.yaml"):
        if doc and doc.get("kind") == "Deployment":
            return doc
    return None


def test_api_runs_as_non_root():
    dep = _api_deployment()
    sc = dep["spec"]["template"]["spec"]["securityContext"]
    assert sc.get("runAsNonRoot") is True


def test_api_container_hardened():
    dep = _api_deployment()
    c = dep["spec"]["template"]["spec"]["containers"][0]
    csc = c["securityContext"]
    assert csc["allowPrivilegeEscalation"] is False
    assert csc["readOnlyRootFilesystem"] is True
    assert "ALL" in csc["capabilities"]["drop"]


def test_api_has_resource_limits():
    dep = _api_deployment()
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["memory"]
    assert c["resources"]["requests"]["cpu"]


def test_api_has_probes():
    dep = _api_deployment()
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert "livenessProbe" in c
    assert "readinessProbe" in c


def test_readonly_root_has_writable_tmp():
    """readOnlyRootFilesystem requires a writable /tmp mount or libs break."""
    dep = _api_deployment()
    spec = dep["spec"]["template"]["spec"]
    mounts = spec["containers"][0]["volumeMounts"]
    assert any(m["mountPath"] == "/tmp" for m in mounts)


def test_all_manifests_valid_yaml():
    """Every manifest parses as valid YAML."""
    for f in K8S.glob("*.yaml"):
        docs = _load_all(f)
        assert all(d is not None for d in docs if d is not None)
