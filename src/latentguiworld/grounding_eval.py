"""Evaluate direct-click or MOVE grounding on five GUI grounding benchmarks."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--mode', choices=['directclick', 'moveto_leftclick'], required=True)
    p.add_argument('--benchmark', action='append', choices=['screenspot-pro', 'screenspot-v2', 'mmbench-gui', 'ui-vision', 'osworld-g'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--initial-cursor-map', type=Path)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--max-model-len', type=int, default=65536)
    a = p.parse_args(argv)
    from gui_agent_captcha.eval import qwen3vl_paired90_five_bench as impl
    from transformers import AutoProcessor
    benchmarks = a.benchmark or list(impl.SUPPORTED_BENCHMARKS)
    processor = AutoProcessor.from_pretrained(str(a.model), trust_remote_code=True) if a.mode == 'directclick' else None
    generator = impl.VLLMGenerator(a.model, tensor_parallel_size=a.tensor_parallel_size,
        max_model_len=a.max_model_len, gpu_memory_utilization=.8,
        image_history_max=1 if a.mode == 'directclick' else 3,
        processor_contract=impl._processor_contract(a.model))
    a.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for benchmark in benchmarks:
        samples = impl.load_benchmark_samples(benchmark, a.data_root, sample_limit=None)
        positions = impl._load_cursor_map(a.initial_cursor_map, {benchmark:samples})[benchmark] if a.initial_cursor_map else None
        with TemporaryDirectory(prefix='grounding-eval-') as temporary:
            opts = dict(generator=generator, output=Path(temporary), checkpoint=a.model,
                        initial_cursor_map=a.initial_cursor_map, cursor_positions=positions)
            summary = impl._run_direct(samples, processor=processor, **opts) if a.mode == 'directclick' else impl._run_moveto(samples, **opts)
            results[benchmark] = {k:v for k,v in summary.items() if k not in {'predictions', 'checkpoint', 'cursor_map_source'} and not ('path' in k or 'file' in k or 'output' in k)}
    (a.output / 'summary.json').write_text(json.dumps({'mode':a.mode,'benchmarks':results}, indent=2)+'\n')
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()
