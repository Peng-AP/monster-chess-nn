import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_search_leaves import signatures


def test_signature_includes_phase_rights_ep_and_budget():
    f=np.zeros((6,840),dtype=np.float32)
    f[1,768]=1;f[2,771]=1;f[3,775]=1;f[4,839]=.5;f[5,839]=.51
    assert len(set(signatures(f)))==6
    assert signatures(f[:1])==signatures(f[:1].copy())


def test_signature_rejects_nonbinary_inputs():
    f=np.zeros((1,840),dtype=np.float32);f[0,3]=.5
    with pytest.raises(ValueError):signatures(f)
