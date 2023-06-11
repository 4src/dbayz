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
#---------------------------------------------
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
     return x

class NUM(COL):
   def __init__(i, txt="",at=0):
     COL.__init__(i,txt=txt,at=at)
     i.w = -1 if len(i.txt) > 0 and i.txt[-1] == "-" else 1
     i.mu = i.m2 = 0
     i.lo, i.hi = big, -big 
   def norm(i,x):
     return x if x=="?" else  (x-i.lo)/(i.hi - i.lo + 1/big)
   def mid(i, decimals=None):
     return rnd(i.mu, decimals)
   def div(i, decimals=None):
     return rnd((i.m2/(i.n - 1))**.5 if i.m2>0 and i.n > 1 else 0, decimals)
   def add1(i,x):
     i.lo = min(x, i.lo)
     i.hi = max(x, i.hi)
     delta = x - i.mu
     i.mu += delta/i.n
     i.m2 += delta*(x - i.mu)

class SYM(COL):
  def __init__(i,txt="",at=0):
    COL.__init__(i,txt=txt,at=at)
    i.counts,i.mode, i.most = {},None,0
  def mid(i,decimals=None): return i.mode
  def div(i, decimals=None):
    a = i.counts
    return rnd( - sum(a[k]/i.n * math.log(a[k]/i.n,2) for k in a if a[k] > 0), decimals)
  def add1(i,x):
    now = i.counts[x] = 1 + i.counts.get(x,0)
    if now > i.most: i.most, i.mode = now, x
  def sub(i,x):
    i.n -= 1
    i.counts[x] -= 1

#---------------------------------------------
class COLS(base):
  def __init__(i,names):
    i.x,i.y, i.names = [],[],names
    i.all = [(NUM if s[0].isupper() else SYM)(at=n,txt=s) for n,s in enumerate(names)]
    for col in i.all:
      z = col.txt[-1]
      if z != "X":
        if z=="!": i.klass= col
        (i.y if z in "-+!" else i.x).append(col)
  def add(i,row):
    for cols in [i.x, i.y]:
      for col in cols: col.add(row.cells[col.at])
    return row
#---------------------------------------------
class DATA(base):
   def __init__(i,src=[]): 
     i.cols, i.rows = None,[]
     [i.add(row) for row in src]
   def add(i,row):
     if i.cols: i.rows += [i.cols.add(row)]
     else:      i.cols = COLS(row.cells)
   def clone(i,rows=[]):
     return DATA([ROW(i.cols.names)] + rows)
   def betters(i,rows=[]):
     return sorted(rows or i.rows, key=cmp_to_key(lambda a,b: i.better(a,b)))
   def better(i,row1,row2):
     s1, s2, n = 0, 0, len(i.cols.y)
     for col in i.cols.y:
       a, b = col.norm(row1.cells[col.at]), col.norm(row2.cells[col.at])
       s1  -= math.exp(col.w * (a - b) / n)
       s2  -= math.exp(col.w * (b - a) / n)
     return s1 / n < s2 / n
#---------------------------------------------
R = random.random        # short cut to random number generator
isa = isinstance         # short cut for checking types
big = 1E30               # large number

def rnd(x,decimals=None):
  return round(x,decimals) if decimals else x

def stats(cols, fun="mid", decimals=2):
  def what(col): return (col.mid if fun=="mid" else col.div)(decimals)
  return dict(mid=BAG(N=cols[1].n, **{col.txt:what(col) for col in cols}))

def rows(file): return csv(file, ROW)

def csv(file,filter=lambda x:x):
  def coerce(x):
    try: return thing(x)
    except: return x
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield filter([coerce(s.strip()) for s in line.split(",")])
#---------------------------------------------
def hEg(): print(__doc__)

def theEg(): print(the)

def rndEg(): assert 3.14 == rnd(math.pi,2)

def numEg(txt=""):
  n = NUM(txt)
  for x in range(10**4):  n.add(R()**.5)
  assert .66 < n.mid() < .67 and .23 <  n.div() < .24
  return n

def symEg(txt=""):
  s=SYM(txt)
  [s.add(x) for x in "aaaabbc"]
  assert "a"==s.mid() and 1.37 <= s.div() < 1.38
  return s

def statsEg():
  print(stats([symEg("sym1"),numEg("num1"),numEg("num2"),symEg("sym2")]))

def rowsEg():
  for row in list(rows(the.file))[:5]: print(row)

def colEg():
  [print(x) for x in COLS(["name","Age","Weight-"]).all]

def dataEg():
  print(stats(DATA(rows(the.file)).cols.y))

def cloneEg():
   d1 = DATA(rows(the.file))
   d2= d1.clone(d1.rows)
   print(d2.cols.y)

def sortEg():
   d = DATA(rows(the.file))
   lst = d.betters()
   m   = int(len(lst)**.5)
   best= d.clone(lst[-m:]); print("best",stats(best.cols.y))
   rest= d.clone(lst[:m]);  print("rest",stats(rest.cols.y))

def okEg():
  saved = {k:v for k,v in the.items()}
  for k,fun in egs.items():
    if k not in ["-ok","-h"]:
      print(colored(k,"yellow",attrs=["bold"]))
      for k,v in saved.items(): the[k] = v
      random.seed(the.seed)
      fun()
#---------------------------------------------
random.seed(the.seed)    # set random number seed

if __name__ == "__main__":
  egs = {("-"+k[:-2]):v for k,v in locals().items() if k[-2:]=="Eg"}
  a=sys.argv[1:]; a and a[0] in egs and egs[a[0]]()
