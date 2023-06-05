#!/usr/bin/env python3 -B
#<!--- vim: set et sts=2 sw=2 ts=2 : --->
"""
## SYNOPSIS:
  fishn: look around just a little, guess where to search.

## USAGE:
  ./fishn.py [OPTIONS] [-g ACTIONS]
  
## DESCRIPTION:
  Use to find best regions within rows of data with multiple objectives.
  N rows of data are ranked via a multi-objective domination predicate
  and then discretized, favoring ranges that distinguish the best
  (N^min) items from a sample of the rest*(N^min)
  
## OPTIONS:
  
     -b  --bins    max number of bins    = 16
     -c  --cohen   size significant separation = .35
     -f  --file    data csv file         = ../data/auto93.csv
     -g  --go      start up action       = nothing
     -h  --help    show help             = False
     -k  --keep    how many nums to keep = 512
     -l  --lazy    lazy mode             = False
     -m  --min     min size              = .5
     -r  --rest    ratio best:rest       = 3
     -s  --seed    random number seed    = 1234567891
     -t  --top     explore top  ranges   = 8
     -w  --want    goal                  = mitigate
"""
import random,math,sys,ast,re
from termcolor import colored
from functools import cmp_to_key

def want(b,r,B,R):
  "We have found `b` of the `B` best rows and 'r' or the `R` rest rows"
  b, r = b/(B + 1/inf), r/(R + 1/inf)
  match the.want:
    case "operate"  : return (b-r)        # want more b than r
    case "mitigate" : return b**2/(b+r)   # want lots of b and far less r
    case "monitor"  : return r**2/(b+r)   # want lots of r and far less b
    case "xtend"    : return 1/(b+r)      # want to go somewhere new
    case "xplore"   : return (b+r)/abs(b - r) # want the decision boundary

the={} 

class obj(object):
  oid = 0
  def __init__(i,**kw): obj.oid+=1; i.__dict__.update(_id=obj.oid, **kw)
  def __repr__(i)     : return printd(i.__dict__)
  def __hash__(i)     : return i._id

def ROW(cells=[]):
  return obj(this=ROW,cells=cells)

def adds(data,row):
  row = ROW(row) if isinstance(row,list) else row # ensure we are reading ROWs
  if not data.cols: # reading row1 (list of column names)
    data.cols = COLS(row.cells)
  else:
    data.rows += [row]
    for cols in [data.cols.x, data.cols.y]:
      for col in cols: add(col,row.cells[col.at])

def COL(at,txt):
  return (NUM if txt[0].isupper() else SYM)(at=at,txt=txt)

def SYM(at=0,txt=""):
  return obj(this=SYM,txt=txt, at=at, n=0,
             counts={}, mode=None, most=0)

def NUM(at=0,txt=""):
   w = -1 if txt and txt[-1]=="-" else 1
   return obj(this=NUM,txt=txt, at=at, n=0, mu=0,m2=0, w=w, lo=inf, hi=-inf)

def add(col,x,n=1):
  if x == "?": return
  col.n += n
  if col.this is SYM:
    now = col.counts[x] = 1 + col.counts.get(x,0)
    if now > col.most: col.most, col.mode = now, x
  else:
    col.lo = min(x, col.lo)
    col.hi = max(x, col.hi)
    delta = x - col.mu
    col.mu += delta/(1.0*col.n)
    col.m2 += delta*(x - col.mu)
  return x

def sd(num):
    return (num.m2/(num.n - 1))**.5 if num.m2>0 and num.n > 1 else 0

def sub(col,x,n=1):
  if x == "?": return
  col.n -= n
  if col.this is SYM:
    col.counts[x] -= n
  else:
    delta = x - col.mu
    col.mu -= delta/(1.0*col.n)
    col.m2 -= delta*(x - col.mu)
  return x

def norm(num,x):
  return x if x=="?" else (x - num.lo) / (num.hi - num.lo + 1/inf)

def mid(col,decimals=None):
  return col.mode if col.this is SYM else rnd(median(ok(col)._kept),decimals)

def div(col,decimals=None):
  return rnd(ent(col.counts) if col.this is SYM else sd(col),decimals)

def stats(data, cols=None, fun=mid, decimals=2):
  return obj(N=len(data.rows), **{c.txt:fun(c,decimals) for c in (cols or data.cols.y)})

def COLS(names):
  cols = obj(this=COLS, names=names, x=[], y=[], 
             all = [COL(n,s) for n,s in enumerate(names)])
  for col in cols.all:
    if col.txt[-1] != "X": 
      (cols.y if col.txt[-1] in "-+!" else cols.x).append(col)
  return cols

def DATA(data=None, src=[]):
  data = data or obj(this=DATA,rows=[], cols=None)
  [adds(data,row) for row in src]
  return data

def clone(data, rows=[]): return DATA(DATA(src=[data.cols.names]), rows)

def betters(data,rows=[]):
  return sorted(rows or data.rows,
                key=cmp_to_key(lambda r1,r2: better(data,r1,r2)))

def better(data, row1, row2):
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a, b = norm(col,row1.cells[col.at]), norm(col,row2.cells[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n
#---------------------------------------------------------------------------------------------------
def tree(best,rest, rows=None,stop=None)
  for row in best: row.klass = True
  for row in rest: row.klass = False
  rows = rows or data.rows
  stop = stop or len(rows)**the.min
  here = {data=clone(data,rows),left=None, right=None, at=None,  val=None}
  _,at,val sort((splitter(data,rows,col) for col in data.cols.x),key=)[0]

def splitter(data,rows,xcol):
  xget = lambda row: row.cells[xcol.at]
  rows = [r for r in rows if xget(r)  != "?"]
  def symSplit():
    syms={}
    for row in rows:
      x = xget(row)
      if x not in syms: syms[x] = SYM(xcol.at,x)
      add(syms[x], row.klass)
    out = sorted(syms.values, key=lambda sym: div(sym))[0]
    return div(out), xcol.at, out.txt
  def numSplit():
    yall,yleft= SYM(),SYM()
    cut   = None
    eps   = div(xcol)*the.cohen
    tiny  = xcol.n**the.min
    xpect = lambda: (yall.n*div(yall) + yleft.n*div(yleft)) / (yall.n + yleft.n)
    [add(yall, row.klass) for row in rows]
    lo = div(yall)
    rows = sorted(rows, key=lambda r: r.cells[col.at]))
    for row in rows:
      add(yleft, sub(yall, row.klass))
      if lhs.n > tiny and rhs.n > tiny:
        x = xget(row)
        if x - xget(rows[0]) >= eps and xget(rows[-1]) -  x >= eps:
          if xpect() < lo:
            cut,lo = x,expect()
    return  lo,col.at,cut
  return numSplit() if col.this is NUM else symSplit()

 def discretize(col,x):
  if x == "?": return
  if col.this is NUM:
    x = int(the.bins*(x - col.lo)/(col.hi - col.lo + 1/inf))
    x = min(the.bins, max(0, x))
  return x



class BIN(object):
  def __init__(self,at=0,txt=" ",lo=None,hi=None,B=0,R=0):
    self.at=at; self.txt=txt; self.lo=lo or inf; self.hi=hi or lo
    self.n=0; self.ys={}; self.score=0; self.B=B; self.R=R
  def __repr__(self):
    if self.hi == self.lo: return f"{self.txt}={self.lo}"
    if s,elf.hi == inf:     return f"{self.txt}>={self.lo}"
    if self.lo == -inf:    return f"{self.txt}<{self.hi}"
    return f"{self.lo} <= {self.txt} < {self.hi}"


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
    for klass in d:
      old = out.ys[klass]  = out.ys.get(klass,set())
      out.ys[klass]  = old | d[klass]
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
  return ent({k:len(set1) for k,set1 in bin.ys.items()})







def contrasts(data1,data2, elite=False):
  top,bins = None, _contrasts(data1,data2)
  n=0
  if elite:
    for bin in sorted(bins,reverse=True,key=lambda bin: bin.score):
      top = top or bin
      if not(bin.lo == -inf and bin.hi == inf) and bin.score > top.score*.1:
        n += 1
        if n <= the.top: yield bin
  else:
    for bin in bins: yield bin



def _contrasts(data1,data2):
  data12 = clone(data1, data1.rows + data2.rows)
  for col in data12.cols.x:
    bins = {}
    for klass,rows in dict(best=data1.rows, rest=data2.rows).items():
      for row in rows:
        x = row.cells[col.at]
        if z := discretize(col, x):
          if z not in bins: bins[z] = BIN(at=col.at,txt=col.txt,lo=x,
                                          B=len(data1.rows), R=len(data2.rows))
          binAdd(bins[z], x, klass, row)
    for bin in merges(col, sorted(bins.values(), key=lambda z:z.lo),"best"):
      yield value(bin, col)


def merges(col,bins,best):
  if col.this is SYM: return bins
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
  b,r = bin.ys.get("best",set()), bin.ys.get("rest",set())
  bin.score = want(len(b),len(r), bin.B, bin.R)
  return bin


def select(bin,row):
  x = row.cells[bin.at]
  if x=="?"                         : return row
  if bin.lo == bin.hi == x          : return row
  if bin.lo == -inf and x <  bin.hi : return row
  if bin.hi ==  inf and x >= bin.lo : return row
  if bin.lo <= x and x < bin.hi     : return row


def selects(bins,rows):
  d={}
  for bin in bins:
    here = d[bin.at] = d.get(bin.at, set())
    d[bin.at] = here | set([row for row in rows if select(bin,row)])
  out= None
  for set1 in d.values():
    if out: out = out & set1
    else  : out = set1
    if len(out)==0: return 
  if len(out) == len(rows): return
  return out

#---------------------------------------------------------------------------------------------------

r=random.random
inf=1E30


def red(s): return colored(s,"red",attrs=["bold"])
def green(s): return colored(s,"green",attrs=["bold"])
def yellow(s): return colored(s,"yellow",attrs=["bold"])
def bold(s): return colored(s,"white",attrs=["bold"])


def rnd(x,decimals=None):
  return round(x,decimals) if decimals else  x


def per(a,p=.5):
  n = max(0, min(len(a)-1, int( 0.5 + p*len(a))))
  return a[n]


def median(a): return per(a,.5)



def stdev(a) : return (per(a,.9) - per(a,.1))/2.56


def ent(a):
  N = sum((a[k] for k in a))
  return - sum(a[k]/N * math.log(a[k]/N,2) for k in a if a[k] > 0)


def powerset(s):
  r = [[]]
  for e in s: r += [x+[e] for x in r]
  return r[1:]


def coerce(x):
  try: return ast.literal_eval(x)
  except: return x


def printd(d):
  p= lambda x: '()' if callable(x) else (f"{x:.2f}" if isinstance(x,float) else str(x))
  return "{"+(" ".join([f":{k} {p(v)}" for k,v in d.items() if k[0]!="_"]))+"}"


def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]


def settings(s):
  setting = r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)"
  return obj(**{m[1]:coerce(m[2]) for m in re.finditer(setting,s)})




def runs():
  if the.help:
    headings, flags ="\n[#]+ [A-Z][A-Z]+:"," [-][-]?[\S]+"
    show  = lambda f: lambda m: f(m[0])
    print(re.sub(headings, show(yellow),re.sub(flags, show(bold), __doc__)))
  else:
    n= sum([run(s,fun) for s,fun in egs.items()
                       if s[0]!="_" and the.go in ["all",s]])
    print(red(f"{n} FAILURE(S)") if n>0 else green(f"{n} FAILURE(S)"))
    return n



def run(s,fun):
  d = the.__dict__
  saved = {k:d[k] for k in d}
  random.seed(the.seed)
  print(bold(s) + " ",end="")
  out = fun()
  for k in saved: d[k] = saved[k]
  print(red("FAIL") if out==False else green("PASS"))
  return out==False


def cli(d):
  d1=d.__dict__
  for k,v in d1.items():
    v = str(v)
    for i,x in enumerate(sys.argv):
      if ("-"+k[0]) == x or ("--"+k) == x:
        d1[k]= coerce("True" if v=="False" else ("False" if v=="True" else sys.argv[i+1]))
  return d



egs={}
def eg(f): egs[f.__name__]= f; return f

@eg
def thed(): print(str(the),"...",end=" ")

@eg
def powered(): print(powerset("abc"),end=" ")

@eg
def colnum():
  num = NUM()
  [add(num, random.gauss(10,2)) for _ in range(the.keep)]
  return 9.95 <= mid(num) <= 10.05 and 1.9 <= div(num) <=2.1

@eg
def colnum2():
  num = NUM()
  the.keep=20
  [add(num, x) for x in range(1000)]
  print(ok(num)._kept)

@eg
def colnum3():
  return the.keep == 512

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
  rows = betters(data)
  print("")
  print(data.cols.names)
  print("all ", stats(data))
  print("best", stats(clone(data,rows[-30:])))
  print("rest", stats(clone(data,rows[:30])))

@eg
def contrasted():
  data = DATA(src=csv(the.file))
  rows = betters(data)
  b = int(len(data.rows)**the.min)
  best = clone(data,rows[-b:])
  rest = clone(data,random.sample(rows, b*the.rest))
  print("\nall ", stats(data))
  print("best", stats(best))
  print("rest", stats(rest))
  b4 = None
  for bin in contrasts(best,rest):
    if bin.txt != b4: print("")
    print(f"{bin.txt:10} {bin.lo:5} {bin.hi:5}",
          obj(best=len(bin.ys.get("best",set())),
              rest=len(bin.ys.get("rest",set())),
              score=f"{bin.score:.2f}"))
    b4 = bin.txt

@eg
def bested():
  data = DATA(src=csv(the.file))
  rows = betters(data)
  b = int(len(data.rows)**the.min)
  best = clone(data,rows[-b:])
  rest = clone(data,random.sample(rows, b*the.rest))
  print("")
  for bin in contrasts(best,rest, elite=True):
    print(bin,bin.score)

@eg
def bested2():
  data = DATA(src=csv(the.file))
  print("\nALL:", stats(data))
  rows = betters(data)
  b = int(len(data.rows)**the.min)
  best = clone(data,rows[-b:])
  rest = clone(data,rows[:b]) #random.sample(rows[:b],b*the.rest))
  print(b*the.rest)
  print("best:", stats(best))
  print("rest:", stats(rest))
  print("")
  tmp,names=[],None
  for bins in powerset(list(contrasts(best,rest, elite=True))):
    if found := list(selects(bins, best.rows + rest.rows)):
      #print([x.cells for x in found])
      d = stats(clone(data, found)).__dict__
      d["Size-"] = len(bins)
      if not names:
        names = list(d.keys()); tmp=[names]
      else:
        row = ROW(list(d.values()))
        row.rule = bins
        tmp += [row]
  rule= betters(DATA(src=tmp))[-1]
  print({s:n for s,n in zip(names,rule.cells)}, rule.rule)


the = settings(__doc__)
if __name__ == "__main__":
  the = cli(the)
  sys.exit(runs())
