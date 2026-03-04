"""Entry point for SNIA comparisons, GIF generation, and single runs.

Usage examples:
  python main.py compare            # run batch comparisons (saves results/*.png)
  python main.py gif                # generate animated GIFs for all problems (saves results/gif/*.gif)
  python main.py run --problem Sphere --algo GA --iter 100
"""

import argparse
import sys


def cmd_compare(args):
    # Delegate to scripts/compare_all.py
    import runpy
    runpy.run_path('scripts/compare_all.py', run_name='__main__')
 

def cmd_gif(args):
    # Delegate to scripts/generate_problem_gifs.py
    import runpy
    runpy.run_path('scripts/generate_problem_gifs.py', run_name='__main__')


def cmd_run(args):
    # Run a single algorithm on a chosen continuous problem and save convergence
    import matplotlib.pyplot as plt

    # Map names to classes
    problems = {
        'Sphere': ('problems.continous.Sphere', 'Sphere'),
        'Rastrigin': ('problems.continous.Rastrigin', 'Rastrigin'),
        'Rosenbrock': ('problems.continous.Rosenbrock', 'Rosenbrock'),
        'Ackley': ('problems.continous.Ackley', 'Ackley'),
        'Griewank': ('problems.continous.Griewank', 'Griewank'),
    }
    algos = {
        'GA': ('algorithms.evolution.GA', 'GeneticAlgorithm'),
        'PSO': ('algorithms.biology.PSO', 'ParticleSwarmOptimization'),
        'DE': ('algorithms.evolution.DE', 'DifferentialEvolution'),
    }

    if args.problem not in problems:
        print('Unknown problem:', args.problem)
        return
    if args.algo not in algos:
        print('Unknown algorithm:', args.algo)
        return

    # dynamic import
    modp, clsnamep = problems[args.problem]
    m = __import__(modp, fromlist=[clsnamep])
    ProbCls = getattr(m, clsnamep)

    moda, clsnamea = algos[args.algo]
    ma = __import__(moda, fromlist=[clsnamea])
    AlgCls = getattr(ma, clsnamea)

    prob = ProbCls(dim=args.dim)
    params = {}
    if args.pop_size:
        params['pop_size'] = args.pop_size

    alg = AlgCls(prob, params=params)
    alg.solve(iterations=args.iter)

    # plot convergence
    vals = [h.get('global_best_fit') for h in alg.history]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(vals) + 1), vals, marker='o')
    plt.title(f'{args.algo} on {args.problem}')
    plt.xlabel('Iteration')
    plt.ylabel('Global Best Fitness')
    out = f'results/{args.problem}_{args.algo}_conv.png'
    plt.grid(alpha=0.25)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved', out)


def build_parser():
    p = argparse.ArgumentParser(prog='main.py')
    sub = p.add_subparsers(dest='cmd')

    sub.add_parser('compare', help='Run batch comparisons and save plots')
    sub.add_parser('gif', help='Generate animated GIFs for all problems (results/gif/)')

    runp = sub.add_parser('run', help='Run one algorithm on a problem')
    runp.add_argument('--problem', default='Sphere')
    runp.add_argument('--algo', default='GA')
    runp.add_argument('--iter', type=int, default=100)
    runp.add_argument('--pop-size', dest='pop_size', type=int, default=40)
    runp.add_argument('--dim', type=int, default=10)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == 'compare':
        cmd_compare(args)
    elif args.cmd == 'gif':
        cmd_gif(args)
    elif args.cmd == 'run':
        cmd_run(args)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
