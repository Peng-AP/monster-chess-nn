"""MCSV002: absolute840 plus piece offsets from White and Black kings.

No rotation/color exchange or prescribed move preferences. Offset coordinates
are signed file/rank differences in [-7,7]. Missing kings contribute no relative
features. Sparse storage avoids materializing the whole 6240-wide corpus on GPU.
"""
import numpy as np
from train_search_value import features as absolute_features

ABSOLUTE_INPUTS=840
RELATIVE_INPUTS=840+2*12*15*15
MAX_ACTIVE=3*64+7


def sparse_features(positions, relative=True):
    base=absolute_features(positions)
    inputs=RELATIVE_INPUTS if relative else ABSOLUTE_INPUTS
    rows,cols=np.nonzero(base)
    values=base[rows,cols]
    if relative:
        piece_rows,piece_cols=np.nonzero(base[:,:768])
        squares=piece_cols%64
        for king_number,channel in enumerate((5,11)):
            kings=base[:,channel*64:(channel+1)*64]
            if (np.count_nonzero(kings,axis=1)>1).any():
                raise ValueError('Multiple kings of one color')
            valid=np.any(kings,axis=1)[piece_rows]
            centers=np.argmax(kings,axis=1)[piece_rows[valid]]
            sq=squares[valid]
            dx=sq%8-centers%8+7
            dy=sq//8-centers//8+7
            extra=840+king_number*2700+(piece_cols[valid]//64)*225+dy*15+dx
            rows=np.concatenate((rows,piece_rows[valid]))
            cols=np.concatenate((cols,extra))
            values=np.concatenate((values,np.ones(len(extra),dtype=np.float32)))
    # Group rows stably. Each active feature is unique, including relative pieces.
    order=np.argsort(rows,kind='stable')
    rows,cols,values=rows[order],cols[order],values[order]
    counts=np.bincount(rows,minlength=len(base))
    if counts.max(initial=0)>MAX_ACTIVE:raise ValueError('Feature bound exceeded')
    starts=np.cumsum(counts)-counts
    slots=np.arange(len(rows))-np.repeat(starts,counts)
    # Distinct padding indices avoid duplicate scatter writes, even for zeros.
    indices=np.broadcast_to(inputs+np.arange(MAX_ACTIVE,dtype=np.int32),
                            (len(base),MAX_ACTIVE)).copy()
    weights=np.zeros(indices.shape,dtype=np.float32)
    indices[rows,slots]=cols
    weights[rows,slots]=values
    return indices,weights


def dense_features(positions, relative=True):
    indices,weights=sparse_features(positions,relative)
    inputs=RELATIVE_INPUTS if relative else ABSOLUTE_INPUTS
    out=np.zeros((len(indices),inputs+MAX_ACTIVE),dtype=np.float32)
    np.put_along_axis(out,indices,weights,axis=1)
    return out[:,:inputs]
