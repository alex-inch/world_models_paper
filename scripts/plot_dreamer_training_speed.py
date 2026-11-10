"""Compare steady-state ARC learning/collection timings from saved SLURM logs."""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ('Original', 'gp8l0xmb', '9202028'),
    ('Renderer fix', 'fh6igknc', '9203250'),
    ('Further optimisations', '9q76bfi4', '9203963'),
    ('BF16', 'f3bqo4fm', '9237753'),
]
TARGET_UPDATES = 2000
TARGET_TRANSITIONS = 8000


def measurements(run, job):
    metadata_file = sorted((ROOT / 'experiments/dreamer/checkpoints' / run).glob('*/metadata.json'))[-1]
    metadata = json.loads(metadata_file.read_text())
    args = metadata['train_args']
    updates = args['collect_interval']
    # Collector logs confirm 8 rollouts, each with 1,000 transitions, in all four runs.
    log = ROOT / f'slurm/slurm-{job}.out'
    text = log.read_text()
    assert 'Collecting 8 rollouts of 1000 transitions' in text
    events = {}
    pattern = r'([\d-]+ [\d:.]+).*Step (\d+): Commencing (data collection|model learning)'
    for stamp, step, phase in re.findall(pattern, text):
        events.setdefault(int(step), {})[phase] = datetime.fromisoformat(stamp)
    samples = []
    checkpoint_every = args.get('steps_per_checkpoint', args['steps_per_eval'])
    for step, event in sorted(events.items()):
        # First two cycles include warmup/compilation. Skip cycles whose end
        # includes evaluation or checkpoint I/O, using the zero-based loop index.
        if step < 3 or (step - 1) % checkpoint_every == 0 or (step - 1) % args['steps_per_eval'] == 0:
            continue
        following = events.get(step + 1, {})
        if 'model learning' not in event or 'data collection' not in following:
            continue
        collection = (event['model learning'] - event['data collection']).total_seconds()
        learning = (following['data collection'] - event['model learning']).total_seconds()
        assert collection > 0 and learning > 0
        samples.append({'outer_step': step, 'collection_seconds': collection,
                        'learning_seconds': learning,
                        'normalised_learning_seconds': learning * TARGET_UPDATES / updates})
    assert samples
    return {'run_id': run, 'job_id': job, 'samples': samples, 'sample_count': len(samples),
            'actual_updates_per_collection': updates,
            'collection_seconds': float(np.mean([s['collection_seconds'] for s in samples])),
            'learning_seconds': float(np.mean([s['normalised_learning_seconds'] for s in samples]))}


def main():
    rows = [dict(label=label, **measurements(run, job)) for label, run, job in RUNS]
    ink, paper = '#343633', '#F7F4EB'
    font = Path('/Users/alexinch/Library/Fonts/routed-gothic.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Routed Gothic', 'font.size': 12,
                         'figure.facecolor': paper, 'axes.facecolor': paper,
                         'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink,
                         'ytick.color': ink, 'svg.fonttype': 'none'})
    # Milliseconds per update at the common ratio of 4 environment transitions
    # per model update: identical work, despite the third run collecting more often.
    learning = np.array([r['learning_seconds'] for r in rows]) / TARGET_UPDATES * 1000
    collection = np.array([r['collection_seconds'] for r in rows]) / TARGET_UPDATES * 1000
    total = learning + collection
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(rows))
    ax.bar(x, learning, width=.58, color=ink, label='MODEL LEARNING', zorder=3)
    ax.bar(x, collection, bottom=learning, width=.58, color='#B4B3AC', label='ENVIRONMENT COLLECTION', zorder=3)
    for i, value in enumerate(total):
        ax.text(i, value + 10, f'{value:.0f} ms', ha='center', va='bottom')
    ax.set_xticks(x, ['Original', '+ Renderer\nfix', '+ Further\noptimisations', '+ BF16'])
    ax.set_ylabel('TIME PER TRAINING STEP (MS)', labelpad=12)
    ax.set_xlabel('1 MODEL UPDATE + 4 ENVIRONMENT TRANSITIONS', labelpad=16, fontsize=10)
    ax.set_ylim(0, total.max() * 1.16)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=ink, alpha=.18, linestyle='--', linewidth=.7)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    fig.tight_layout()
    out = ROOT / 'assets/arc-training-speed'
    for extension in ('.png', '.svg', '.pdf'):
        fig.savefig(out.with_suffix(extension), dpi=300, bbox_inches='tight')
    payload = {'normalisation': '2000 model updates + 8000 environment transitions; plotted as ms per model update + 4 transitions',
               'hardware': 'All four runs: one NVIDIA L40S, batch size 32, sequence length 64, eight environments.',
               'method': 'Mean phase durations from SLURM timestamps; first two cycles and evaluation/checkpoint cycles excluded. Collection includes policy inference, simulation, IPC and replay storage. Learning includes metric logging. Third run uses 1600 updates per collection: learner duration scaled by 2000/1600. Observational comparison, not a controlled ablation.',
               'overall_speedup': float(total[0] / total[-1]), 'runs': rows}
    out.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
    with out.with_suffix('.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'run_id', 'cycles', 'learning_ms', 'collection_ms', 'total_ms', 'speedup_vs_original'])
        for i, row in enumerate(rows):
            writer.writerow([row['label'], row['run_id'], row['sample_count'], learning[i], collection[i], total[i], total[0] / total[i]])
    print(out.with_suffix('.png'))
    print(out.with_suffix('.csv').read_text())


if __name__ == '__main__':
    main()
