"""Tests for GitHub auth and private registry helpers in the plugins browser."""

from __future__ import annotations

from io import BytesIO
import subprocess
from unittest.mock import patch

import pytest
import urllib.error
import yaml

from avlite.plugins.p60_visualizer_tk import p63_plugins_app as cp


@pytest.fixture
def token_path(tmp_path, monkeypatch):
    path = tmp_path / "github_oauth.json"
    monkeypatch.setattr(cp._GitHubClient, "_github_token_path", staticmethod(lambda: path))
    return path


def test_git_auth_args_empty():
    assert cp._GitOperations._git_auth_args(None) == []
    assert cp._GitOperations._git_auth_args("") == []


def test_git_auth_args_bearer():
    args = cp._GitOperations._git_auth_args("secret-token")
    assert args[0] == "-c"
    assert "Authorization: Basic" in args[1]
    assert "Bearer" not in args[1]


def test_git_subprocess_env():
    assert cp._GitOperations._git_subprocess_env()["GIT_TERMINAL_PROMPT"] == "0"


def test_authenticated_clone_url():
    url = cp._GitOperations._authenticated_clone_url("https://github.com/AV-Lab/a2rl.git", "tok")
    assert url == "https://x-access-token:tok@github.com/AV-Lab/a2rl.git"


def test_save_and_load_github_token(token_path):
    with patch.object(cp._GitHubClient, "_github_user", return_value="avlab-user"):
        cp._GitHubClient._save_github_token("tok123", "avlab-user")
        assert token_path.stat().st_mode & 0o777 == 0o600
        loaded = cp._GitHubClient._load_github_token()
    assert loaded == ("tok123", "avlab-user")


def test_clear_github_token(token_path):
    token_path.write_text("{}", encoding="utf-8")
    cp._GitHubClient._clear_github_token()
    assert not token_path.exists()


def test_fetch_registry_public():
    payload = yaml.safe_dump({"plugins": [{"name": "demo", "repository": "https://github.com/x/y"}]})
    response = BytesIO(payload.encode())

    class FakeResp:
        def read(self):
            return response.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        plugins = cp._PluginOperations.fetch_registry(private=False)
    assert plugins[0]["name"] == "demo"


def test_fetch_registry_private():
    payload = yaml.safe_dump({"plugins": [{"name": "private_plugin", "repository": "https://github.com/a/b"}]})
    with patch.object(cp._GitHubClient, "_github_api", return_value=payload.encode()) as api_mock:
        plugins = cp._PluginOperations.fetch_registry(private=True, token="gho_test")
    assert plugins[0]["name"] == "private_plugin"
    api_mock.assert_called_once()
    assert cp.PRIVATE_REGISTRY_REPO in api_mock.call_args[0][0]
    assert "ref=main" in api_mock.call_args[0][0]


def test_parse_github_sso_url():
    header = (
        "required; url=https://github.com/orgs/AV-Lab/sso"
        "?authorization_request=ABC123"
    )
    assert cp._GitHubClient._parse_github_sso_url(header) == (
        "https://github.com/orgs/AV-Lab/sso?authorization_request=ABC123"
    )
    assert cp._GitHubClient._parse_github_sso_url("") is None


def test_github_api_raises_with_sso_url():
    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                "https://api.github.com/test",
                403,
                "Forbidden",
                {
                    "X-GitHub-SSO": (
                        "required; url=https://github.com/orgs/AV-Lab/sso"
                        "?authorization_request=ABC123"
                    )
                },
                BytesIO(
                    b'{"message":"Resource protected by organization SAML enforcement."}'
                ),
            )

    with patch("urllib.request.urlopen", side_effect=FakeHTTPError()):
        with pytest.raises(cp.GitHubApiError) as exc_info:
            cp._GitHubClient._github_api("https://api.github.com/test", "gho_test")
    err = exc_info.value
    assert err.status == 403
    assert "SAML" in str(err)
    assert err.sso_url == "https://github.com/orgs/AV-Lab/sso?authorization_request=ABC123"


def test_poll_device_flow_returns_token():
    responses = [
        {"error": "authorization_pending"},
        {"access_token": "gho_new", "token_type": "bearer"},
    ]

    def fake_post(url, data):
        return responses.pop(0)

    with patch.object(cp._GitHubClient, "_github_form_post", side_effect=fake_post):
        with patch.object(cp.time, "sleep"):
            token = cp._GitHubClient._poll_device_flow("device-code", interval=1, expires_in=60)
    assert token == "gho_new"


def test_start_device_flow_requires_client_id(monkeypatch):
    monkeypatch.setattr(cp, "GITHUB_OAUTH_CLIENT_ID", "")
    with pytest.raises(ValueError, match="AVLITE_GITHUB_OAUTH_CLIENT_ID"):
        cp._GitHubClient._start_device_flow()


def test_install_plugin_uses_git_auth(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    entry = {
        "name": "p",
        "repository": "https://github.com/org/private-plugin",
        "version": "latest",
    }
    cp._PluginOperations.install_plugin(entry, tmp_path, token="gho_test")
    clone_cmd = calls[0]
    assert clone_cmd[0] == "git"
    assert clone_cmd[1] == "clone"
    assert "x-access-token:gho_test@github.com/org/private-plugin" in clone_cmd[4]
    set_url_cmd = next(c for c in calls if "set-url" in c)
    assert set_url_cmd[-1] == "https://github.com/org/private-plugin"


def test_notify_host_changed_calls_on_community_plugins_changed():
    class FakeHost:
        def __init__(self):
            self.called = False

        def on_community_plugins_changed(self):
            self.called = True

    host = FakeHost()
    panel = object.__new__(cp._PluginRegistryPanel)
    panel._host = host
    cp._PluginRegistryPanel._notify_host_changed(panel)
    assert host.called


def test_notify_host_changed_no_op_without_host():
    panel = object.__new__(cp._PluginRegistryPanel)
    panel._host = None
    cp._PluginRegistryPanel._notify_host_changed(panel)
