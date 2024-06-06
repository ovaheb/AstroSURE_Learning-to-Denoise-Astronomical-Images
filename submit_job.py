#!/usr/bin/env python

from datetime import datetime
import argparse
import shlex
import os

exbl_all = ['echo hostname: `hostname`']

c_time = '-d time=10:00'
c_hold = '-d nodes=1 time=3:00:00 -e "sleep 99d  # hold the node so can ssh seperately"'
c_gres = '-d gres=gpu:1'
c_smi = '-e "echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" -e nvidia-smi'
c_gpu = f'{c_gres} {c_smi}'


pre_dirv = {
'email': ['mail-user=omid.vaheb@mail.utoronto.ca', 'mail-type=ALL'],
}

pre_exbl = {
'exp_omp': ['export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK'],
}


templates = {

'raw': '',

'hold':
f'-d ntasks=1 {c_hold}',

'ghold':
f'-d mem=0 cpus-per-task=24 {c_gpu} {c_hold}',

'single':
f'-d nodes=1 ntasks-per-node=1 {c_time}',

'gsingle':
f'-d mem=0 cpus-per-task=24 {c_time} {c_gpu}',

'mpi':
f'-d nodes=2 ntasks-per-node=1 time=00:16:00',

'gmpi':
f'-d nodes=2 ntasks-per-node=1 cpus-per-task=6 mem=0 {c_time} {c_gpu}',

'graham':
f'-d nodes=1 gres=gpu:2 ntasks-per-node=32 mem=127000M {c_time} {c_smi}',

'cedarP1004':
f'-d nodes=1 gres=gpu:p100:4 ntasks-per-node=4 cpus-per-task=6 exclusive mem=125G {c_time} {c_smi}',
# ntasks-per-node doesn't spawn 24 tasks unless srun is used.
# is this to make slurm believe node is used in full efficiency?

'cedarP100L4':
f'-d nodes=1 gres=gpu:p100l:4 ntasks-per-node=4 cpus-per-task=6 mem=0 {c_time} {c_smi}',
# req. only whole nodes

'narval':
f'-d nodes=1 gres=gpu:1 cpus-per-task=10 mem=32G export=ALL',
}


def create(dirvs, exbls):
    content = []
    w = lambda s_: content.append(s_)
    d = lambda s_: w(f'#SBATCH --{s_}')
    e = lambda s_: w(f'echo "{s_}"')

    w('#!/bin/bash%s'%(' -i' if _a.interactive else ''))
    d('account=rrg-kyi')
    d('time=15:00:00')
    for it in dirvs: d(it)
    e('BEGIN EXECUTABLES')
    for it in exbls: w(it)
    e('END EXECUTABLES')
    return '\n'.join(content)

def main():
    _t = make_base_parser().parse_args(shlex.split(templates[_a.template]))
    # process directives
    _key = lambda d_: d_.split('=')[0]
    same_dirv = lambda d1,d2: _key(d1)==_key(d2)
    already_in = lambda d1,dirv: any(same_dirv(d1,d2) for d2 in dirv)
    def rm_duplicates(dirv):
        new_dirv = []
        for d_ in reversed(dirv):   # last one takes precedence
            if not already_in(d_, new_dirv):
                new_dirv.append(d_)
        return list(reversed(new_dirv))

    _t.dirv, _a.dirv = rm_duplicates(_t.dirv), rm_duplicates(_a.dirv)
    dirvs, added = [], []
    for tem in _t.dirv:
        dirvs.append(tem)
        for arg in _a.dirv:
            if same_dirv(arg,tem):
                dirvs[-1] = arg     # override the directive in template
                added.append(arg)

    for arg in _a.dirv:
        if arg not in added:
            dirvs.append(arg)       # add user defined directives

    aexbl = _a.exbl if _a.prefix is None else [f'{_a.prefix} {line}' for line in _a.exbl]
    exbls = [*exbl_all, *_t.exbl, *aexbl]
    text = create(dirvs, exbls)
    if _a.print: print(text)
    else: sbatch(text)

def sbatch(text):
    print('='*50, text, '='*50, sep='\n')
    JOBS_PATH = os.path.join(os.environ['SCRATCH'], 'slurm_jobs')
    if not os.path.exists(JOBS_PATH): os.makedirs(JOBS_PATH)
    time_str = datetime.utcnow().strftime('%Y-%m-%d--%H-%M-%S.%f')[:-3]
    job_name = _a.name if _a.name else _a.template
    file_name = f'job_{job_name}_{time_str}'
    file_path = os.path.join(JOBS_PATH, file_name+'.sh')
    with open(file_path, 'w') as fp: fp.write(text)

    out_pattern = f'{file_name}_%J.out'
    cmd_str = f'sbatch --output {out_pattern} --chdir {JOBS_PATH} {file_path}'
    print(cmd_str)
    os.system(cmd_str)
    if _a.watch: os.system('watch -n 4 squeue -u $USER')


class AppendAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        def pre(dd, dest):
            for val in values:
                getattr(args, dest).extend(dd[val])
        def new(dest): getattr(args, dest).extend(values)

        if self.dest=='p_dirv': pre(pre_dirv, 'dirv')
        elif self.dest=='p_exbl': pre(pre_exbl, 'exbl')
        elif self.dest=='dirv': new('dirv')
        elif self.dest=='exbl': new('exbl')
        else: raise argparse.ArgumentTypeError(f'Unexpected dest:{self.dest}')


def make_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirv', help='overridable directives', nargs='+', action=AppendAction, default=[])
    parser.add_argument('-e', '--exbl', help='executable lines, e.g. echo "running"', nargs='+', action=AppendAction, default=[])
    return parser

def make_parser():
    parser = make_base_parser()
    parser.add_argument('template', choices=templates.keys())
    parser.add_argument('-D', '--p_dirv', help='pre-specified directives', nargs='+', action=AppendAction, choices=list(pre_dirv.keys()))
    parser.add_argument('-E', '--p_exbl', help='pre-specified executables', nargs='+', action=AppendAction, choices=list(pre_exbl.keys()))
    parser.add_argument('--prefix', help='prefix all executable lines with this', type=str, default=None)
    parser.add_argument('--name', help='job name, output file prefix', type=str)
    parser.add_argument('--interactive', help='add -i to bash shebang, expands aliases', action='store_true')
    parser.add_argument('--print', help='only print output to console', action='store_true')
    parser.add_argument('--watch', help='watch squeue after submission', action='store_true')
    return parser

if __name__ == '__main__':
    _a = make_parser().parse_args()
    main()

    