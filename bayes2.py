#!/usr/bin/env python3 -B#<!-- vim: set ts=2 sw=2 et: -->
"""
SYNOPSIS: 
    bayes2: look around just a little, then find some good stuff

USAGE:
    ./bayes2.py [OPTIONS] [-g ACTIONS]

DESCRIPTION:
    Use to find best regions within rows of data with multiple objectives.
    N rows of data are ranked via a multi-objective domination predicate
    and then discretized, favoring ranges that distinguish the best
    (N^min) items from a sample of the rest*(N^min)

OPTIONS:
    -b  --bins    max number of bins                     = 16
    -B  --Bootstraps number of bootstrap samples         = 512
    -C  --Cohen   'not different' if under the.cohen*sd  = .35
    -c  --cliffs  Cliff's Delta limit                    = .147
    -d  --distance2Far                                   = .92
    -f  --file    data csv file                          = ../data/auto93.csv
    -F  --Fars    how far is far away?                   = 256
    -g  --go      start up action                        = nothing
    -h  --help    show help                              = False
    -m  --min     on N items, recurse down to N**min     = .5
    -n  --n       explore all subsets of top ''n bins    = 7
    -p  --p       distance exponent                      = 2
    -r  --rest    expand to (N**min)**rest               = 4
    -s  --seed    random number seed                     = 1234567891
    -S  --Some    max items kept in Some                 = 256
    -w  --want    mitigate,operate,monitor,xplore,xtend  = mitigate

COPYRIGHT:
    (c) 2023, Tim Menzies, <timm@ieee.org>  BSD-2
"""
import random,math,sys,ast,re
from copy import deepcopy
from termcolor import colored
from functools import cmp_to_key as cmp2key

def main(help):
  if help:
    def bold(m):   return colored(first(m), attrs=["bold"])
    def bright(m): return colored(first(m), "light_yellow")
    print(re.sub("\n[A-Z][A-Z]+:", bold, re.sub(" [-][-]?[\S]+", bright, help)))
  else:
    n  = sum([run(the,fun,name) for fun,name in egs if (the.go=="." or the.go==name)])
    yell("green" if n==0 else "red", f"# overall, FAILS = {n}\n")
    return n

#----------------------------------------------------
class obj(object):
  oid = 0
  def __init__(self,**d):  obj.oid+=1; self.__dict__.update(oid=obj.oid,**d)
  def __hash__(self): return self.oid
  def __repr__(self):
    d = self.__dict__.items()
    return "{"+(" ".join([f":{k} {nice(v)}" for k,v in d if first(k)!="_"]))+"}"

def THE(s):
  return obj(**{m[1]:coerce(m[2]) for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",s)})

def COL(txt=" ",  at=0, data=None):
  col = (NUM if first(txt).isupper() else SYM)(txt=txt,at=at)
  if data and last(txt) != "X":
    data.all += [col]
    (data.y if last(txt) in "-+" else data.x).append(col)
  return col

def SYM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, has={}, mode=None,most=0,isNum=False)

def NUM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, mu=0, m2=0, sd=0, isNum=True,
             lo=inf, hi=- inf, w= -1 if last(txt)=="-" else 1)

def DATA(data=None, src=[]):
  data = data or obj(rows=[], names=[], all=[], x=[], y=[])
  for row in src:
    if data.names:
      for col in data.all: add(col,row[col.at])
      data.rows += [row]
    else:
      data.names = row
      for at,txt in enumerate(row): COL(txt,at,data)
  return data

def clone(data, rows=[]):
  return DATA(data=DATA(src=[data.names]),src=rows)

def adds(col,lst=[]): 
  for x in lst: add(col,x)
  return col

def add(col,x,inc=1):
  if x == "?": return
  col.n  += inc
  if col.isNum:
    col.lo  = min(col.lo, x)
    col.hi  = max(col.hi, x)
    d       = x - col.mu
    col.mu += d/col.n
    col.m2 += d*(x - col.mu)
    col.sd  = 0 if col.n<2 else (col.m2/(col.n - 1))**.5
  else:
    tmp = col.has[x] = inc + col.has.get(x,0)
    if tmp >  col.most: col.most, col.mode = tmp,x

def stats(data,mid=True,cols=None):
  def fun(col)      : return (middle if mid else diversity)(col)
  def rnd(n)        : return round(n, ndigits=3) if isinstance(n,(float,int)) else n
  def middle(col)   : return col.mu if col.isNum else col.mode
  def diversity(col): return col.sd if col.isNum else ent(col.has)
  return obj(N=len(data.rows), **{col.txt:rnd(fun(col)) for col in (cols or data.y)})

#----------------------------------------------------
def around(data,row, rows=None):
  return sorted([(dist(data,row,r),r) for r in (rows or data.rows)],key=first)

def dist(data,row1,row2):
  d = sum((aha(col,row1[col.at],row2[col.at])**the.p for col in data.x))
  return (d/len(data.x))**(1/the.p)

def aha(col,x,y):
  if x=="?" and y=="?": return 1
  if col.isNum:
    if   x=="?" : y=norm(col,y); x=1 if y<.5 else 0
    elif y=="?" : x=norm(col,x); y=1 if x<.5 else 0
    else        : x,y = norm(col,x), norm(col,y)
    return abs(x-y)
  else:
    return 0 if x==y else 1

def norm(col,x):
  return (x - col.lo) / (col.hi - col.lo + 1/inf)

def polarize(data,rows):
  random.shuffle(rows)
  some     = rows[:the.Fars]
  far      = int(the.distance2Far * len(some))
  north    = around(data, first(some), some)[far]
  south    = around(data, north,       some)[far]
  gap      = lambda r1,r2: dist(data,r1,r2)
  c        = gap(north,south)
  project  = lambda r: (gap(north,r)**2 + c**2 - gap(south,r)**2)/(2*c)
  rows     = sorted(((project(r),r) for r in rows), key=first)
  mid      = len(rows)//2
  return [r[1] for r in rows[:mid]], [r[1] for r in rows[mid:]],north,south,c

def elite(data,rows,stop=None,rest=[]):
  stop = stop or len(rows)**the.min
  if stop <= len(rows):
    return rows, random.sample(rest, int(len(rows)*the.rest))
  else:
    norths,souths,north,south = polarize(data,rows)
    if better(data,south,north): norths,souths = souths,norths
    return elite(data,norths, stop=stop, rest=rest + souths)

#----------------------------------------------------
def ordered(data,rows=[]):
  return sorted(rows or data.rows,
                key=cmp2key(lambda r1,r2: better(data,r1,r2)))

def better(data,row1,row2):
  s1, s2, n = 0, 0, len(data.y)
  for col in data.y:
    a,b  = norm(col,row1[col.at]), norm(col,row2[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

#---------------------------------------------------------------------------------------------------
def BIN(at=0,txt=" ",lo=None,hi=None,B=0,R=0):
  return obj(at=at, txt=txt, lo=lo or inf, hi=hi or lo,
             n=0, ys={}, score=0, B=B, R=R)

def binAdd(bin, x, y, row):
  bin.n     += 1
  bin.lo     = min(bin.lo, x)
  bin.hi     = max(bin.hi, x)
  bin.ys[y]  = bin.ys.get(y,[]) 
  bin.ys[y] += [row]
  return bin

def binEnt(bin):
  def p(n): return n*math.log(n,2)
  return -sum((p(len(lst)/bin.n) for lst in bin.ys.values()))

def merge(bin1, bin2):
  """Merge two adjacent bins."""
  out = BIN(at=bin1.at, txt=bin1.txt,
            lo=bin1.lo, hi=bin2.hi,
            B=bin1.B, R=bin2.R)
  out.n = bin1.n + bin2.n
  for d in [bin1.ys, bin2.ys]:
    for key in d:
      out.ys[key]  = out.ys.get(key,[])
      out.ys[key] += d[key]
  return out

def merged(bin1,bin2,num):
  out   = merge(bin1,bin2)
  eps   = num.sd*the.Cohen
  small = num.n / the.bins
  if bin1.n <= small or bin1.hi - bin1.lo < eps : return out
  if bin2.n <= small or bin2.hi - bin2.lo < eps : return out
  e1, e2, e3 = binEnt(bin1), binEnt(bin2), binEnt(out)
  if e3 < (bin1.n*e1 + bin2.n*e2)/out.n : return out

#---------------------------------------------------------------------------------------------------
def discretize(col,x):
  if x == "?": return
  if col.isNum:
    lo, hi = col.mu - 2*col.sd, col.mu + 2*col.sd
    x = int(the.bins*(x - lo)/(hi - lo + 1/inf))
  return x

def contrasts(data1,data2):
  data12 = clone(data1, data1.rows + data2.rows)
  for col in data12.x:
    bins = {}
    for klass,rows in dict(best=data1.rows, rest=data2.rows).items():
      for row in rows:
        x = row[col.at]
        if z := discretize(col, x):
          if z not in bins: bins[z] = BIN(at=col.at,txt=col.txt,lo=x,
                                          B=len(data1.rows), R=len(data2.rows))
          binAdd(bins[z], x, klass, row)
    for bin in merges(col, sorted(bins.values(), key=lambda z:z.lo)):
      yield value(bin, col)

def value(bin,col):
  b,r = bin.ys.get("best",[]), bin.ys.get("rest",[])
  bin.score = want(len(b),len(r), bin.B, bin.R)
  return bin

def want(b,r,B,R):
  b, r = b/(B + 1/inf), r/(R + 1/inf)
  match the.want:
    case "operate":  return (b-r)
    case "mitigate": return b**2/(b+r)
    case "monitor":  return r**2/(b+r)
    case "xtend":    return 1/(b+r)
    case "xplore":   return (b+r)/abs(b - r)

def merges(col,bins):
  if not col.isNum: return bins
  bins = mergeds(bins,col)
  for j in range(len(bins)-1): bins[j].hi = bins[j+1].lo
  first(bins).lo = -inf
  last(bins).hi =  inf
  return bins

def mergeds(a, col):
  b,j = [],0
  while j < len(a):
    now = a[j]
    if j < len(a) - 1:
      if new := merged(a[j], a[j+1], col): now,j = new,j+1
    b += [now]
    j += 1
  return a if len(a) == len(b) else mergeds(b, col)

#---------------------------------------------------------------------------------------------------
def rules(data1,data2):
  bins = [bin for bin in contrast(data1,data2)]
  bins = sorted(bins, key=lambda z:z.score, reverse=True)[:8]
  print([bin.score for bin in bins])
  for rule in powerset(bins):
    evaluate(rule)
  return bins

#---------------------------------------------------------------------------------------------------
inf = 1E60

def first(a): return a[0]
def last(a): return a[-1]

def cli(d):
  for k,v in d.items():
    v= str(v)
    for i,x in enumerate(sys.argv):
      if ("-"+first(k)) == x or ("--"+k) == x:
        v= "False" if v=="True" else ("True" if v=="False" else sys.argv[i+1])
    d[k] = coerce(v)
  return d

def yell(c,*s):
  print(colored(''.join(s),"light_"+c,attrs=["bold"]),end="")

def ent(d):
  N = sum([d[k] for k in d])
  return - sum((n/N*math.log(n/N,2) for n in d.values()))

def prin(*l): print(*l,end="")

def nice(x):
  if callable(x)         : return x.__name__+'()'
  if isinstance(x,float) : return f"{x:.2f}"
  return x

def coerce(x):
  try: return ast.literal_eval(x)
  except: return x

def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]

def sample(a):
  return random.choices(a,k=len(a))

def powerset(s):
  r = [[]]
  for e in s: r += [x+[e] for x in r]
  return r

def different(x,y):
  return cliffsDelta(x,y) and bootstrap(x,y)

def cliffsDelta(x,y):
  n = 0
  if len(x) > 10*len(y) : x = random.choices(x,10*len(y))
  if len(y) > 10*len(x) : y = random.choices(y,10*len(x))
  lt,gt = 0,0
  for x1 in x:
    for y1 in y:
      n = n + 1
      if x1 > y1: gt = gt + 1
      if x1 < y1: lt = lt + 1
  return abs(lt - gt)/n > the.Cliffs # true if different

def bootstrap(x,t,conf=.05):
  delta= lambda x,y: abs(x.mu-y.mu) / ((x.sd**2/x.n + y.sd**2/y.n)**.5 + 1/inf)
  x, y, z = NUM(), NUM(), NUM()
  for y1 in y0: add(x,y1); add(z,y1)
  for z1 in z0: add(x,z1); add(z,z1)
  yhat = [y1 - y.mu + x.mu for y1 in y0]
  zhat = [z1 - z.mu + x.mu for z1 in z0]
  d    = delta(y,z)
  n    = 0
  for _ in range(the.Bootstraps):
    ynum = adds(NUM(), sample(yhat))
    znum = adds(NUM(), sample(zhat))
    if delta(ynum, znum)  > d:
      n += 1
  return n / the.Bootstraps < conf # true if different

def run(the,fun,name):
  yell("yellow","# ",name)
  random.seed(the.seed)
  b4  = deepcopy(the)
  tmp = fun()
  for k,v in b4.__dict__.items(): the.__dict__[k] = v
  yell("red"," FAIL\n") if tmp==False else yell("green"," PASS\n")
  return tmp==False
#---------------------------------------------------------------------------------------------------

egs=[]
def eg(f): global egs; egs += [(f,f.__name__)]; return f

@eg
def they(): the.p=23; prin("",str(the)[:40],"...")

@eg
def andThen(): return the.p==2

@eg
def power():
  print([x for x in powerset([1,2,3])],end=" ")
  return 2**3  == len([x for x in powerset([1 ,2,3])])

@eg
def numed():
  num = adds(NUM(), [random.random() for _ in range(10**3)])
  return .28 < num.sd < .3 and .49 < num.mu < .51

@eg
def symed():
  sym = adds(SYM(), "aaaabbc")
  return 1.37 < ent(sym.has) < 1.39 and sym.mode == 'a'

@eg
def csvd():
  return 3192 == sum((len(a) for a in csv(the.file)))

@eg
def dists():
  data = DATA(src=csv(the.file))
  for row in data.rows:
    d = dist(data, row, first(data.rows))
    if not (0 <= d <= 1): return False

@eg
def arounds():
  data = DATA(src=csv(the.file))
  print("0.000",first(data.rows))
  for d,r in around(data,first(data.rows))[::50]:
    print(f"{d:.3f}",r)

@eg
def sorter():
  data = DATA(src=csv(the.file))
  rows = ordered(data)
  print("")
  print(data.names)
  print("all ", stats(data))
  print("best", stats(clone(data,rows[-30:])))
  print("rest", stats(clone(data,rows[:-30])))

@eg
def const():
  data = DATA(src=csv(the.file))
  rows = ordered(data)
  b = int(len(data.rows)**the.min)
  r = b*the.rest
  best = clone(data,rows[-b:])
  rest = clone(data,random.sample(rows[:-b],r))
  print("\nall ", stats(data))
  print("best", stats(best))
  print("rest", stats(rest))
  b4=None
  for bin in contrasts(best,rest):
    if bin.txt != b4: print("")
    print(bin.txt, bin.lo, bin.hi,
          {k:len(bin.ys[k]) for k in sorted(bin.ys)},
          f"{bin.score:.2f}")
    b4 = bin.txt

#------------------------------------------------------------------------------
the = THE(__doc__)
if __name__ == "__main__":
  the.__dict__ = cli(the.__dict__)
  main(__doc__ if the.help else "")
