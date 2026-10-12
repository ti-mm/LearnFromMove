"""Check effective training settings, frozen scenes, and evaluation output handling."""
import json
from pathlib import Path
from collections import Counter
import pytest
from latentguiworld.train import ROOT, RECIPES, materialize
from latentguiworld.evaluate import SUITE, VARIANTS, evaluate
from gui_agent_captcha.benchmarks.exploration_depth.contracts import PAPER_VARIANTS


def test_frozen_suite():
    m=json.loads(SUITE.read_text())
    assert m['suite_id']=='learn_from_move' and m['frozen']
    assert Counter(e['variant'] for e in m['episodes']) == dict.fromkeys(PAPER_VARIANTS,150)
    assert all(e['viewport']==[1280,720] for e in m['episodes'])
    for e in m['episodes']:
        assert (SUITE.parent / e['canonical_case_path']).is_file()


@pytest.mark.parametrize('recipe',RECIPES)
def test_effective_hydra_settings(recipe,tmp_path):
    from hydra import compose,initialize_config_dir
    plan=materialize(recipe,model='models/model',train_files=['data/train.parquet'],output=tmp_path)
    cmd=plan['command']; i=cmd.index('verl.trainer.sft_trainer')+1 if plan['kind']=='sft' else 3
    with initialize_config_dir(config_dir=str(ROOT/'third_party/verl/verl/trainer/config'),version_base=None):
        c=compose(config_name='sft_trainer_engine' if plan['kind']=='sft' else 'ppo_trainer',overrides=cmd[i:])
    assert c.trainer.total_training_steps is None
    assert c.data.train_files==[str((ROOT/'data/train.parquet').resolve())]
    if plan['kind']=='sft':
        assert c.data.micro_batch_size_per_gpu==1 and c.optim.lr==3e-6
        assert c.trainer.total_epochs==(3 if recipe=='benchmark_sft' else 2)
        assert c.data.train_batch_size==(64 if recipe=='benchmark_sft' else 128)
    else:
        a=c.actor_rollout_ref.actor; r=c.actor_rollout_ref.rollout
        assert c.trainer.total_epochs==1 and r.temperature==1 and r.top_p==.95
        assert a.ppo_micro_batch_size_per_gpu==1 and a.ppo_epochs==1
        assert not c.algorithm.use_kl_in_reward
        if recipe.startswith('grounding'):
            assert c.data.train_batch_size==c.data.gen_batch_size==8 and r.n==8
            assert a.optim.lr==5e-7 and a.kl_loss_coef==.01
        else:
            assert c.data.train_batch_size==c.data.gen_batch_size==16 and r.n==5
            assert a.ppo_mini_batch_size*r.n==80
            assert a.optim.lr==1e-6 and a.kl_loss_coef==.005
            assert all(loop['max_generation_tokens']==384 for loop in plan['agent_loops'][:3])


def test_grounding_mask():
    from gui_agent_captcha.train.verl_groundcua_window_dataset import apply_target_only_loss_mask
    assert apply_target_only_loss_mask([0,1,1],[1,1,0],trainable=True)==[0,1,0]
    assert apply_target_only_loss_mask([0,1,1],[1,1,0],trainable=False)==[0,0,0]


class ProbeBackend:
    checkpoint_path=Path("probe")
    _call_index=0
    max_new_tokens=128
    call_log_dir=None
    last_raw_prediction=None
    last_think_text=None
    def predict_action(self,obs,history,**kwargs):
        from gui_agent_captcha.actions import PrimitiveAction
        self._call_index+=1
        if not history: return PrimitiveAction(kind='move_to',x=500,y=800)
        if len(history)==1: return PrimitiveAction(kind='mouse_down')
        return PrimitiveAction(kind='mouse_up')


@pytest.mark.browser
def test_evaluation_six_environment_plumbing(tmp_path):
    result=evaluate(ProbeBackend(),output=tmp_path,limit=1)
    assert result['episodes']==6 and not result['full_suite']
    assert result['infrastructure_errors']==0
    assert set(p.name for p in tmp_path.iterdir())=={'summary.json',*(v+'.json' for v in VARIANTS)}
    assert all(s['episodes']==1 for s in result['variants'].values())
