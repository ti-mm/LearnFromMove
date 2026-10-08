"""Materialize paper settings and launch SFT or GRPO."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ('benchmark_sft', 'grounding_directclick_sft', 'grounding_moveto_leftclick_sft', 'drag_rl', 'ten_choice_rl', 'rotation_rl',
           'grounding_directclick_rl', 'grounding_moveto_leftclick_rl')


def loops(output: Path):
    common = {'max_steps': 12, 'max_infra_retries': 2, 'context_window': 16384,
              'generation_safety_reserve_tokens': 32, 'max_generation_tokens': 384,
              'artifact_root': str(output / 'rollouts')}
    entries = []
    for view, cls in [('first', 'First'), ('third', 'Third')]:
        entries.append(dict(common, name=f'exploration_depth_{view}_person_with_think_online_v1',
            _target_=f'gui_agent_captcha.benchmarks.exploration_depth.{view}_person_online_agent_loop.ExplorationDepth{cls}PersonWithThinkAgentLoopV1',
            storage_root=str(ROOT), browser_slot_dir=str(output / 'browser_slots')))
    entries.append(dict(common, name='paired_rotation_with_think_browser_online_v1',
        _target_='gui_agent_captcha.train.paired_rotation_browser_agent_loop_v1.PairedRotationWithThinkBrowserAgentLoopV1'))
    for track in ['directclick', 'moveto_leftclick']:
        entries.append({'name': f'groundcua_paper_style_{track}_v1',
            '_target_': 'gui_agent_captcha.train.groundcua_paper_style_agent_loop.GroundCUAPaperStyleAgentLoop',
            'artifact_root': str(output / 'rollouts'), 'max_response_tokens': 256})
    return entries


def materialize(recipe: str, *, model: str, train_files: list[str], output: Path,
                gpus: int = 8, nodes: int | None = None, node_rank: int = 0,
                val_files: list[str] | None = None, master_addr: str = '127.0.0.1', master_port: int = 29500, save_freq: int = 20):
    config = json.loads((ROOT / 'configs' / f'{recipe}.json').read_text())
    nodes = nodes if nodes is not None else config.get('default_nodes', 1)
    output = output.resolve()
    overrides = {k.lstrip('+'): v for k, v in config['overrides'].items()}
    overrides.update({'trainer.default_local_dir': str(output / 'checkpoints'),
        'trainer.n_gpus_per_node': gpus, 'trainer.nnodes': nodes,
        'trainer.project_name': 'latentguiworld', 'trainer.experiment_name': recipe,
        'trainer.save_freq': save_freq, 'data.train_files': [str(Path(f).resolve()) for f in train_files]})
    if config['kind'] == 'sft':
        overrides['model.path'] = str(Path(model).resolve())
        if overrides['data.train_batch_size'] % (gpus * nodes):
            raise ValueError('SFT global batch must be divisible by the GPU world size')
        command = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={gpus}', f'--nnodes={nodes}']
        command += ['--standalone'] if nodes == 1 else [f'--node_rank={node_rank}', f'--master_addr={master_addr}', f'--master_port={master_port}']
        command += ['-m', config['entrypoint'], 'engine=fsdp', 'optim=fsdp']
    else:
        # VERL constructs this loader even when periodic validation is disabled.
        overrides['data.val_files'] = [str(Path(f).resolve()) for f in (val_files or train_files)]
        samples = overrides['actor_rollout_ref.actor.ppo_mini_batch_size'] * overrides['actor_rollout_ref.rollout.n']
        if samples % (gpus * nodes):
            raise ValueError(f'{samples} action samples must be divisible by the GPU world size; use 8 or 16 GPUs for paired RL')
        overrides.update({'actor_rollout_ref.model.path': str(Path(model).resolve()),
            'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(output / 'agent_loops.yaml')})
        command = [sys.executable, '-m', config['entrypoint']]
    def resolve(value):
        if isinstance(value, str): return value.replace('${ROOT}', str(ROOT))
        if isinstance(value, list): return [resolve(v) for v in value]
        if isinstance(value, dict): return {k: resolve(v) for k,v in value.items()}
        return value
    overrides = {k: resolve(v) for k,v in overrides.items()}
    command += ['++' + k + '=' + json.dumps(v, separators=(',', ':')) for k,v in overrides.items()]
    command += ['hydra.run.dir=' + str(output / 'hydra'), 'hydra.output_subdir=null']
    return {'recipe': recipe, 'kind': config['kind'], 'command': command, 'overrides': overrides,
            'dataset': config['dataset'], 'agent_loops': loops(output)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--model', required=True, help='Local Hugging Face model directory')
    p.add_argument('--train-files', nargs='+', required=True)
    p.add_argument('--val-files', nargs='+', help='Separate validation Parquet files')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--nodes', type=int)
    p.add_argument('--node-rank', type=int, default=0)
    p.add_argument('--master-addr', default='127.0.0.1')
    p.add_argument('--master-port', type=int, default=29500)
    p.add_argument('--save-freq', type=int, default=20)
    p.add_argument('--dry-run', action='store_true', help='Write resolved configuration and command')
    a = p.parse_args(argv)
    if min(a.gpus, a.nodes or 1, a.save_freq) < 1: p.error('GPU, node, and save frequency values must be positive')
    plan = materialize(a.recipe, model=a.model, train_files=a.train_files, output=a.output,
        gpus=a.gpus, nodes=a.nodes, node_rank=a.node_rank, val_files=a.val_files, master_addr=a.master_addr,
        master_port=a.master_port, save_freq=a.save_freq)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'resolved_config.json').write_text(json.dumps(plan, indent=2) + '\n')
    # JSON is valid YAML and preserves strings exactly.
    (a.output / 'agent_loops.yaml').write_text(json.dumps(plan['agent_loops'], indent=2) + '\n')
    print(json.dumps(plan, indent=2))
    if a.dry_run: return 0
    import pyarrow.parquet as pq
    rows = sum(pq.read_metadata(f).num_rows for f in a.train_files)
    counts = plan['dataset']
    expected = counts.get('prompts', counts.get('examples'))
    if rows != expected:
        raise ValueError(f'{a.recipe} requires {expected} rows, received {rows}')
    if not Path(a.model).is_dir(): raise FileNotFoundError(a.model)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'third_party/verl'), env.get('PYTHONPATH','')])
    env['GUI_CAPTCHA_STORAGE_ROOT'] = str(ROOT)
    env['IMAGE_MAX_PIXELS'] = '921600'
    return subprocess.call(plan['command'], cwd=ROOT, env=env)

if __name__ == '__main__':
    raise SystemExit(main())
