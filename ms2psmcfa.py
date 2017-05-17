import sys
import itertools as it

L = int(sys.argv[1])
rl = lambda: sys.stdin.readline().strip()

def skip():
    l = sys.stdin.readline()
    if not l:
        raise StopIteration
    while not l.strip():
        if not l:
            raise StopIteration
        l = sys.stdin.readline()
    return l.strip()

rl()
rl()

def slices(iterable, size):
    it = iter(iterable)
    item = list(it.islice(it, size))
    while item:
        yield item
        item = list(it.islice(it, size))

try:
    for i in it.count():
        skip()
        n = int(rl().split()[1])
        s = ['T'] * n
        ps = [int(float(x) * L) for x in rl().split()[1:]]
        for p, x, y in zip(ps, rl(), rl()):
            if x != y:
                s[p] = 'K'
        print('>{}'.format(i))
        print('\n'.join(''.join(x) for x in (s[j:min(j+60,len(s))] for j in range(0, len(s), 60))))
except StopIteration:
    pass
