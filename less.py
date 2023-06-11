#!/usr/bin/env python3 -B
#<!--- vim: set et sts=2 sw=2 ts=2 : --->
"""
less: look around just a little, guess where to search.
(c) 2023 Tim Menzies <timm@ieee.org>, BSD-2

USAGE: ./less.py [OPTIONS] 

OPTIONS:

  -b  --bins    max number of bins    = 16
  -c  --cohen   size significant separation = .35
  -f  --file    data csv file         = "../data/auto93.csv"
  -h  --help    show help             = False
  -k  --keep    how many nums to keep = 256
  -l  --lazy    lazy mode             = False
  -m  --min     min size              = .5
  -r  --rest    ratio best:rest       = 3
  -s  --seed    random number seed    = 1234567891
  -t  --top     explore top  ranges   = 8
  -w  --want    goal                  = "mitigate"
"""
import random,math,sys,ast,re
from termcolor import colored
from functools import cmp_to_key
from ast import literal_eval as thing

class BAG(dict): __getattr__ = dict.get

the = BAG(**{m[1]:thing(m[2])
          for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",__doc__)})

random.seed(the.seed)    # set random number seed
R = random.random        # short cut to random number generator
isa = isinstance         # short cut for checking types
big = 1E30               # large number

class base(object):
   def __repr__(i): 
     return i.__class__.__name__+str({k:v for k,v in i.__dict__.items() if k[0] != "_"})

class ROW(base):
   def __init__(i, cells=[]): i.cells=cells

class COL(base):
   def __init__(i, txt="",at=0): i.n,i.at,i.txt = 0,at,txt
   def add(i,x):
     if x != "?":
        i.n += 1
        i.add1(x)

def rnd(x,decimals=None):
  return round(x,decimals) if decimals else x

def per(a,p=.5):
  return a[int(max(0,min(len(a)-1,p*len(a))))]

class NUM(COL):
   def __init__(i, txt="",at=0):
     COL.__init__(i,txt=txt,at=at)
     i.w = -1 if len(i.txt) > 0 and i.txt[-1] == "-" else 1
     i._has,i.ready = [],False
     i.lo, i.hi = big, -big 
   def has(i):
      if not i.ready:
         i.ready=True
         i._has.sort()
         i.lo,i.hi = i._has[0], i._has[-1]
      return i._has
   def norm(i,x):
     return x if x=="?" else  (x-i.lo)/(x.hi - x.lo + 1/big)
   def mid(i,decimals=None):
     return rnd( per(i.has(),.5), decimals)
   def div(i,decimals=None):
     return rnd( (per(i.has(),.9) - per(i.has(),.1))/2.56, decimals)
   def add1(i,x):
     a = i._has
     if   len(a) < the.keep  : i.ready=False; a += [x]
     elif R() < the.keep/i.n : i.ready=False; a[ int(len(a)*R()) ] = x
   def sub1(i,x): raise(DeprecationWarning("sub not defined for NUMs"))

class SYM(COL):
  def __init__(i,txt="",at=0):
    COL.__init__(i,txt=txt,at=at)
    i.counts,i.mode, i.most = {},None,0
  def mid(i,**_): return i.mode
  def div(i, decimals=None):
    a = i.counts
    return rnd( - sum(a[k]/i.n * math.log(a[k]/i.n,2) for k in a if a[k] > 0), decimals)
  def add1(i,x):
    now = i.counts[x] = 1 + i.counts.get(x,0)
    if now > i.most: i.most, i.mode = now, x
  def sub(i,x):
    i.n -= 1
    i.counts[x] -= 1

def stats(cols, fun="mid", decimals=2):
  fun = lambda i,d: i.mid(d) if fun=="mid" else i.div(d)
  return dict(mid=BAG(N=cols[1].n, **{col.txt:fun(col,decimals) for col in cols}))

class COLS(base):
  def __init__(i,names):
    i.x,i,y, i.names = names,[],[]
    i.all = [(NUM if s[0].isupper() else SYM)(s,n) for n,s in enumerate(names)]
    for col in i.all:
      z = col.txt[-1]
      if z != "X":
        if z=="!": i.klass= col
        (i.y if z in "-+!" else i.y).append(col)
  def add(i,row):
    for cols in [i.x, i.y]:
      for col in cols: col.add(row.cells[col.at])
    return row

def csv(file,filter=lambda x:x):
  def coerce(x):
    try: return thing(x)
    except: return x
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield filter([coerce(s.strip()) for s in line.split(",")])

class DATA(base):
   def __init__(i,src=[]): 
     i.cols, i.rows = None,[]
     [i.add(row) for row in src]
   def add(i,row):
     row = ROW(row) if isa(row,list) else row
     if i.cols:
       i.rows += [i.cols.add(row)]
     else:
       i.cols = COLS(row.cells)
   def clone(i,rows=[]):
     return DATA([i.cols.names] + rows)
   def sort(i,rows=[]):
     return sorted(rows or i.rows, key=cmp_to_key(lambda r1,r2: i.better(r1,r2)))
   def better(i,row1,row2):
     s1, s2, n = 0, 0, len(i.cols.y)
     for col in i.cols.y:
       a, b = col.norm(row1.cells[col.at]), col.norm(row2.cells[col.at])
       s1  -= math.exp(col.w * (a - b) / n)
       s2  -= math.exp(col.w * (b - a) / n)
     return s1 / n < s2 / n

egs={}                                  # place to store examples
def eg(f): egs["-"+f.__name__[:-2]]= f; return f # define one example

@eg
def hEg(): print(__doc__)

@eg
def theEg(): print(the)

@eg
def rndEg(): assert 3.14 == rnd(math.pi,2)

@eg
def perEg(): assert 33 == per([i for i in range(100)], .33)

@eg
def numEg(txt=""):
  for keep in [16,32,64,128,256,512,1024,2048]:
    the.keep = keep
    n = NUM(txt)
    for i in range(10**3):  n.add(i)
    print(keep,BAG(mid=n.mid(),midError=int(100*(n.mid()-500)/500), 
                   div=n.div(), divError=int(100*(n.div()-288)/288)))
  return n

@eg
def symEg(txt=""):
  s=SYM(txt)
  [s.add(x) for x in "aaaabbc"]
  assert "a"==s.mid() and 1.37 <= s.div() < 1.38
  return s

@eg
def statsEg():
  print(stats([symEg("sym1"),numEg("num1"),symEg("sym2")]))

@eg
def okEg():
  reset = {k:v for k,v in the.items()}
  for k,fun in egs.items():
    if k not in ["-ok","-h"]:
      random.seed(the.seed)
      fun()
      for k,v in reset.items(): the[k] = v

if __name__ == "__main__":
  a=sys.argv; a[1:] and a[1] in egs and egs[a[1]]()
