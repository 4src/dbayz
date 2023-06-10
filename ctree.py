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
def tree(data,best,rest, stop=None):
  for row in best.rows: row.klass = True
  for row in rest.rows: row.klass = False
  rows = best.rows + rest.rows
  data1 = clone(data, rows)
  return tree1(data, rows, len(rows)**the.min)

def tree1(data, rows,stop):
  node = {data=clone(data,rows), how=None, leftFun=None, left=None, right=None}
  if len(rows) > stop:
    _,_,cut,how,leftFun  =  sort((splitter(here,col) for col in data.cols.x),key=)[0]
    if cut:
      here.how, here.leftFun = how,leftFun
      left,right = [],[]
      for row in rows:
        (left if  leftFun(row) else right).append(row)
      if len(left)  < len(rows):  here.left  = tree1(data,left,stop)
      if len(right) < len(rows):  here.right = tree1(data,right,stop)
  return here

def splitter(data,rows,xcol):
  return (numSplit if col.this is NUM else symSplit)(data,rows,xcol)

def symSplit(data,rows,xcol):
  syms={}
  for row in rows:
    x = row.cells[xcol.at]
    if x != "?":
      if x not in syms: syms[x] = SYM(at=xcol.at, txt=x)
      add(syms[x], row.klass)
  out = sorted(syms.values, key=lambda sym: div(sym))[0]
  return (div(out),  xcol.at, out.txt, f"{xcol.name} = {out.txt}", 
          lambda r:r.cells[xcol.at] in ["?",out.txt])

def numSplit(data,rows,xcol):
  eps  = div(xcol)*the.cohen
  tiny = xcol.n**the.min
  xget = lambda r:r.cells[xcol.at]
  rows = sorted([row for row in rows if xget(row)  != "?"], key=xget)
  yall,yleft= SYM(),SYM()
  [add(yall, row.klass) for row in rows]
  cut, lo  = cut, div(yall)
  for row in rows:
    add(yleft, sub(yall, row.klass))
    if lhs.n > tiny and rhs.n > tiny:
      x = xget(row)
      if x - xget(rows[0]) >= eps and xget(rows[-1]) -  x >= eps:
        tmp  = (yall.n*div(yall) + yleft.n*div(yleft)) / (yall.n + yleft.n)
        if tmp < lo:
          cut,lo = x,tmp
  return (lo, xcol.at, cut, f"{xcol.name} <= {cut}",
          lambda r: xget(r) == "?" or xget(r) <= cut)

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
