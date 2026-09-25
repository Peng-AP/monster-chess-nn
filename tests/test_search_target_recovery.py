"""Receipt-lock recovery must never silently admit changed playing inputs."""
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import match_evidence
from search_targets_campaign import recovery_changes


def test_atomic_replace_retries_permission_error(tmp_path,monkeypatch):
    path=tmp_path/'status.json';match_evidence.atomic_json(path,{'old':True})
    replace=match_evidence.os.replace;calls=[];sleeps=[]
    def locked(a,b):
        calls.append(1)
        if len(calls)<3:raise PermissionError('reader still owns old file')
        replace(a,b)
    monkeypatch.setattr(match_evidence.os,'replace',locked)
    monkeypatch.setattr(match_evidence.time,'sleep',sleeps.append)
    match_evidence.atomic_json(path,{'new':True})
    assert json.loads(path.read_text())=={'new':True}
    assert len(calls)==3 and sleeps==[.05,.05]


def test_atomic_replace_permanent_lock_preserves_previous(tmp_path,monkeypatch):
    path=tmp_path/'status.json';match_evidence.atomic_json(path,{'old':True})
    calls=[]
    def locked(a,b):
        calls.append(1);raise PermissionError('permanent')
    monkeypatch.setattr(match_evidence.os,'replace',locked)
    monkeypatch.setattr(match_evidence.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):match_evidence.atomic_json(path,{'new':True})
    assert len(calls)==21 and json.loads(path.read_text())=={'old':True}


def test_recovery_only_allows_receipt_and_driver_changes():
    receipt=str(ROOT/'src/match_evidence.py');engine=str(ROOT/'native/monster_native.pyd')
    before={receipt:'old',engine:'same'};after={receipt:'new',engine:'same'}
    assert set(recovery_changes(before,after))=={receipt}
    with pytest.raises(ValueError):recovery_changes(before,{**after,engine:'changed'})
    with pytest.raises(ValueError):recovery_changes(before,{receipt:'new'})


@pytest.mark.skipif(sys.platform!='win32',reason='Windows delete-sharing semantics')
def test_atomic_replace_survives_real_windows_reader(tmp_path):
    import ctypes
    from ctypes import wintypes
    import threading
    import time
    path=tmp_path/'status.json';match_evidence.atomic_json(path,{'old':True})
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
        wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,0,None)
    assert handle!=wintypes.HANDLE(-1).value
    def release():
        time.sleep(.15);kernel.CloseHandle(handle)
    reader=threading.Thread(target=release);reader.start()
    try:match_evidence.atomic_json(path,{'new':True})
    finally:reader.join()
    assert json.loads(path.read_text())=={'new':True}
