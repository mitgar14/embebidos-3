"""Tests del módulo shared pid_utils."""
import os

from pid_utils import is_pid_alive, check_cmdline


def test_is_pid_alive_self():
    assert is_pid_alive(os.getpid()) is True


def test_is_pid_alive_unlikely_pid():
    # PID 99999999 muy improbable de existir
    assert is_pid_alive(99999999) is False


def test_is_pid_alive_string_pid():
    assert is_pid_alive(str(os.getpid())) is True


def test_check_cmdline_returns_false_on_missing_marker():
    # Pytest no contiene 'nano_build_engine' en su cmdline
    assert check_cmdline(os.getpid()) is False


def test_check_cmdline_invalid_pid():
    assert check_cmdline(99999999) is False
