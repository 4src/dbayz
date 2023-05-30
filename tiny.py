# vim: set et sts=2 sw=2 ts=2 :
"""
tiny: look around just a little, then find some good stuff
(c) 2023, Tim Menzies, <timm@ieee.org>  BSD-2

USAGE: ./tiny.py [OPTIONS] [-g ACTIONS]

DESCRIPTION:
  Use to find best regions within rows of data with multiple objectives.
  N rows of data are ranked via a multi-objective domination predicate
  and then discretized, favoring ranges that distinguish the best
  (N^min) items from a sample of the rest*(N^min)

OPTIONS:

     -b  --bins    max number of bins    = 16  
     -c  --cohen   size significant separation = .35  
     -f  --file    data csv file         = ../data/auto93.csv  
     -g  --go      start up action       = nothing  
     -h  --help    show help             = False  
     -k  --keep    how many nums to keep = 512
     -m  --min     min size              = .5
     -r  --rest    ratio best:rest       = 3
     -s  --seed    random number seed    = 1234567891  
     -w  --want    what goal to chase    = mitigate
"""
import random,math,sys,ast,re
from termcolor import colored
from functools import cmp_to_key

the={} # global options, filled in later

def isa(x,y): return isinstance(x,obj) and x.isa == y

def SYM(at=0,txt=""):
  return obj(isa=SYM,txt=txt, at=at, n=0,
             counts={}, mode=None, most=0)

def NUM(at=0,txt=""):
   w = -1 if txt and txt[-1]=="-" else 1
   return obj(isa=NUM,txt=txt, at=at, n=0,
              _kept=[], ok=True, w=w, lo=inf, hi=-inf)

def norm(num,x):
  return x if x=="?" else (x-num.lo) / (num.hi - num.lo + 1/inf)

def add(col,x,n=1):
  if x == "?": return
  col.n += n
  if isa(col,NUM):
    col.lo = min(x, col.lo)
    col.hi = max(x, col.hi)
    a = col._kept
    if   len(a) < the.keep    : col.ok=False; a += [x]
    elif r() < the.keep/col.n : col.ok=False; a[int(len(a)*r())] = x
  else:
    now = col.counts[x] = 1 + col.counts.get(x,0)
    if now > col.most: col.most, col.mode = now, x

def ok(col):
  if isa(col,NUM) and not col.ok:
    col._kept.sort()
    col.ok=True
  return col

def div(col,decimals=None):
  return rnd(stdev(ok(col)._kept) if isa(col,NUM) else ent(col.counts), decimals)

def mid(col,decimals=None):
  return rnd(median(ok(col)._kept),decimals) if isa(col,NUM) else col.mode

#---------------------------------------------------------------------------------------------------
def ROW(cells=[]):
  return obj(isa=ROW,cells=cells)

def COLS(names):
  cols = obj(isa=COLS,names=None, x=[], y=[], all=[])
  cols.names = names
  for n,s in enumerate(names):
    col = (NUM if s[0].isupper() else SYM)(at=n,txt=s)
    cols.all += [col]
    if s[-1] != "X":
      (cols.y if s[-1] in "-+!" else cols.x).append(col)
  return cols

def DATA(data=None, src=[], filter=lambda x:x):
  data = data or obj(isa=DATA,rows=[], cols=None)
  for row in src:
    if not data.cols:
      data.cols = COLS(row)
    else:
      lst = row.cells if isa(row,ROW) else row
      for cols in [data.cols.x, data.cols.y]:
        for col in cols:
          x = lst[col.at] = filter(lst[col.at])
          add(col,x)
      data.rows += [row]
  return data

def clone(data, rows=[]): return DATA(DATA(src=[data.cols.names]), rows)

def ordered(data,rows=[]):
  return sorted(rows or data.rows,
                key=cmp2key(lambda r1,r2: better(data,r1,r2)))


def stats(data, cols=None, fun=mid, decimals=2):
  tmp = {col.txt:fun(col,decimals) for col in (cols or data.cols.y)}
  return obj(N=len(data.rows),**tmp)

def better(data, row1, row2):
  "`Row1` is better than `row2` if moving to it losses less than otherwise."
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a, b = norm(col,row1[col.at]), norm(col,row2[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

def ordered(data,rows=[]):
  return sorted(rows or data.rows,
                key=cmp_to_key(lambda r1,r2: better(data,r1,r2)))

#---------------------------------------------------------------------------------------------------
def discretize(col,x):
  if x == "?": return
  if isa(col,NUM):
    x = int(the.bins*(x - col.lo)/(col.hi - col.lo + 1/inf))
    x = min(the.bins, max(0, x))
  return x

def BIN(at=0,txt=" ",lo=None,hi=None,B=0,R=0):
  return obj(at=at, txt=txt, lo=lo or inf, hi=hi or lo,
             n=0, ys={}, score=0, B=B, R=R)

def binAdd(bin, x, y, row):
  bin.n     += 1
  bin.lo     = min(bin.lo, x)
  bin.hi     = max(bin.hi, x)
  bin.ys[y]  = bin.ys.get(y,set()) 
  bin.ys[y].add(row)
  return bin

def merge(bin1, bin2):
  out = BIN(at=bin1.at, txt=bin1.txt,
            lo=bin1.lo, hi=bin2.hi,
            B=bin1.B, R=bin2.R)
  out.n = bin1.n + bin2.n
  for d in [bin1.ys, bin2.ys]:
    for key in d:
      old = out.ys[key]  = out.ys.get(key,set())
      out.ys[key]  = old | d[key]
  return out

def merged(bin1,bin2,num,best):
  out   = merge(bin1,bin2)
  eps   = div(num)*the.cohen
  small = num.n / the.bins
  if len(out.ys.get(best,[]))/out.B < 0.05: return out
  if bin1.n <= small or bin1.hi - bin1.lo < eps : return out
  if bin2.n <= small or bin2.hi - bin2.lo < eps : return out
  e1, e2, e3 = binEnt(bin1), binEnt(bin2), binEnt(out)
  if e3 <= (bin1.n*e1 + bin2.n*e2)/out.n : return out

def binEnt(bin):
  return ent({k:len(lst) for k,lst in bin.ys.items()})

def contrasts(data1,data2):
  data12 = clone(data1, data1.rows + data2.rows)
  for col in data12.cols.x:
    bins = {}
    for klass,rows in dict(best=data1.rows, rest=data2.rows).items():
      for row in rows:
        x = row[col.at]
        if z := discretize(col, x):
          if z not in bins: bins[z] = BIN(at=col.at,txt=col.txt,lo=x,
                                          B=len(data1.rows), R=len(data2.rows))
          binAdd(bins[z], x, klass, row)
    for bin in merges(col, sorted(bins.values(), key=lambda z:z.lo),"best"):
      yield value(bin, col)

def merges(col,bins,best):
  if not isa(col,NUM): return bins
  bins = mergeds(bins,col,best)
  for j in range(len(bins)-1): bins[j].hi = bins[j+1].lo
  bins[0].lo = -inf
  bins[-1].hi =  inf
  return bins

def mergeds(a, col,best):
  b,j = [],0
  while j < len(a):
    now = a[j]
    if j < len(a) - 1:
      if new := merged(a[j], a[j+1], col,best): now,j = new,j+1
    b += [now]
    j += 1
  return a if len(a) == len(b) else mergeds(b, col,best)

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

#---------------------------------------------------------------------------------------------------
inf=float("inf")
class obj(object):
  oid = 0
  def __init__(i,**kw): obj.oid+=1; i.__dict__.update(_id=obj.oid, **kw)
  def __repr__(i)     : return printd(i.__dict__)
  def __hash__(i)     : return i._id

r=random.random

def rnd(x,decimals=None):
   return round(x,decimals) if decimals else  x

def per(a,p=.5):
  n = max(0, min(len(a)-1, int( p*len(a))))
  return a[n]

def median(a): return per(a,.5)
def stdev(a) : return (per(a,.9) - per(a,.1))/2.56
def ent(a):
  N = sum((a[k] for k in a))
  return - sum(a[k]/N * math.log(a[k]/N,2) for k in a if a[k] > 0)

def coerce(x):
  try: return ast.literal_eval(x)
  except: return x

def printd(d):
  p= lambda x: '()' if callable(x) else (f"{x:.2f}" if isinstance(x,float) else str(x))
  return "{"+(" ".join([f":{k} {p(v)}" for k,v in d.items() if k[0]!="_"]))+"}"

def cli(d):
  d1=d.__dict__
  for k,v in d1.items():
    v = str(v)
    for i,x in enumerate(sys.argv):
      if ("-"+k[0]) == x or ("--"+k) == x:
        d1[k]= coerce("True" if v=="False" else ("False" if v=="True" else sys.argv[i+1]))
  return d

def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]

def settings(s):
  return obj(**{m[1]:coerce(m[2])
                for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",s)})

def run(s,fun):
  d = the.__dict__
  saved = {k:d[k] for k in d}
  random.seed(the.seed)
  out = fun()
  for k in saved: d[k] = saved[k]
  color = "red" if out==False else "green"
  print(colored(f"FAIL {s}" if out==False else f"PASS {s}",color))
  return out==False
#---------------------------------------------------------------------------------------------------
egs={}
def eg(f): egs[f.__name__]= f; return f

@eg
def thed(): print(str(the)[:50],"... ")

@eg
def colnum():
  num = NUM()
  [add(num,x) for x in range(20)]
  return 0==num.lo and 19==num.hi

@eg
def colnum2():
  num = NUM()
  [add(num,r()) for x in range(10**4)]
  return .28 < div(num) < .32 and .46 < mid(num) < .54

@eg
def colsym():
  sym = SYM()
  [add(sym,c) for c in "aaaabbc"]
  return 1.37 < div(sym) < 1.38 and mid(sym)=='a'

@eg
def csved(): return 3192==sum((len(a) for a in csv(the.file)))

@eg
def statd():
  data=DATA(src=csv("../data/auto93.csv"))
  print(stats(data,cols=data.cols.x))
  print(stats(data))
  print(stats(data,fun=div))

@eg
def orderer():
  data = DATA(src=csv(the.file))
  rows = ordered(data)
  print("")
  print(data.cols.names)
  print("all ", stats(data))
  print("best", stats(clone(data,rows[-30:])))
  print("rest", stats(clone(data,rows[:30])))

@eg
def contraster():
  data = DATA(src=csv(the.file))
  rows = ordered(data)
  b = int(len(data.rows)**the.min)
  r = b*the.rest
  best = clone(data,rows[-b:])
  #rest = clone(data,random.sample(rows[:-b],r))
  rest = clone(data,rows[:r])
  print("\nall ", stats(data))
  print("best", stats(best))
  print("rest", stats(rest))
  b4 = None
  for bin in contrasts(best,rest):
    if bin.txt != b4: print("")
    print(bin.txt, bin.lo, bin.hi,
          {k:len(bin.ys[k]) for k in sorted(bin.ys)}, f"{bin.score:.2f}")
    b4 = bin.txt

#---------------------------------------------------------------------------------------------------
the = settings(__doc__)
if __name__ == "__main__":
  the = cli(the)
  if the.help: print(__doc__)
  else: sys.exit(sum([run(s,fun) 
                      for s,fun in egs.items() if s[0]!="_" and the.go in ["all",s]]))
