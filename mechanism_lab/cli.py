import argparse
import json
from pathlib import Path
from .core import Task, simulate, compare, recover, intervention

def main():
    parser = argparse.ArgumentParser(description='Mechanism Lab experiments')
    parser.add_argument('command', choices=['simulate','compare','recover','intervene'])
    parser.add_argument('--model', default='qlearning')
    parser.add_argument('--trials', type=int, default=160)
    parser.add_argument('--reversal', type=int, default=80)
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args()
    task = Task(trials=args.trials, reversal=args.reversal)
    if args.command == 'simulate': result = simulate(task, args.model)
    elif args.command == 'compare': result = compare(simulate(task, args.model))
    elif args.command == 'recover': result = recover(args.model, task=task)
    else: result = intervention(task, args.model, changed=(.6, 5.) if args.model == 'qlearning' else None)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f'Wrote {args.output}')

if __name__ == '__main__': main()
