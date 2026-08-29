# tests/conftest.py
import os
import pathlib

_here = pathlib.Path(__file__).resolve()

# The kit's serve/ profile exists only in the private working repo. In a public
# checkout there is no serve/ directory, so pointing at it made every test error
# with "Profile not found" at import. Prefer it when present; otherwise use the
# profile shipped INSIDE lens_kit, which exists in every install.
_serve = _here.parents[3] / "serve" / "profiles" / "qwen-serve.yaml"
if _serve.exists():
    _profile = _serve
else:
    from lens_kit import builtin_profile_path

    _profile = builtin_profile_path()

os.environ.setdefault("LENS_HOOK_PROFILE", str(_profile))
