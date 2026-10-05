import numpy as np
from experiments.common import History, potential
from experiments.data import generate_case, stream_case


def test_potential_gradient_and_tempering():
    rng=np.random.default_rng(231)
    h=History(3,block_size=2)
    for x,y in zip(rng.normal(size=(5,3)),[1,-1,1,1,-1]):h.append(x,y)
    th=rng.normal(size=(2,3));x=rng.normal(size=3);p=np.array([.2,.7])
    val,grad=potential(th,2.,h,extra_x=x,extra_p=p,last_weight=.3)
    for i in range(2):
        for j in range(3):
            e=np.zeros_like(th);e[i,j]=1e-6
            vp,_=potential(th+e,2.,h,extra_x=x,extra_p=p,last_weight=.3)
            vm,_=potential(th-e,2.,h,extra_x=x,extra_p=p,last_weight=.3)
            assert abs((vp[i]-vm[i])/2e-6-grad[i,j])<1e-7


def test_datasets_fixed_and_bounded(tmp_path):
    for family in ['signal','isotropic','separable','rare_flip','correlated','coordinate','shift','random']:
        case=dict(d=5,B=100.,T=261,B_generator=3.,family=family,data_seed=413)
        a=generate_case(tmp_path/family/'a',case)
        b=generate_case(tmp_path/family/'b',case)
        assert a['sha256_interleaved_blocks']==b['sha256_interleaved_blocks']
        batches=list(stream_case(tmp_path/family/'a',5,261))
        x=np.concatenate([z[0] for z in batches]);y=np.concatenate([z[1] for z in batches])
        assert len(x)==261 and np.max(np.linalg.norm(x,axis=1))<=1+2e-16
        assert set(y)<=set([-1,1])
