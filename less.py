#!/usr/bin/env python3 -B
#<!--- vim: set et sts=2 sw=2 ts=2 : --->
"""
## SYNOPSIS:
  less: look around just a little, guess where to search.

## USAGE:
  ./less.py [OPTIONS] [-g ACTIONS]

## DESCRIPTION:
  Use to find best regions within rows of data with multiple objectives.
  N rows of data are ranked via a multi-objective domination predicate
  and then discretized, favoring ranges that distinguish the best
  (N^min) items from a sample of the rest*(N^min)

## OPTIONS:

     -b  --bins    max number of bins    = 16
     -c  --cohen   size significant separation = .35
     -f  --file    data csv file         = "../data/auto93.csv"
     -g  --go      start up action       = "nothing"
     -h  --help    show help             = False
     -k  --keep    how many nums to keep = 512
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

class BAG(dict): __getattr__ = dict.get
the = BAG(**{m[1]:ast.literal_eval(m[2])
           for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",__doc__)})

big = 1E30
R = random.random
isa = isinstance
egs={}
def eg(f): egs[f.__name__]= f; return f
def run1(): egs[sys.argv[1]]()

class base(object):
   def __repr__(i): 
     return i.__class__.__name__+str({k:v for k,v in i.__dict__.items() if k[0] != "_"})

class ROW(base):
   def __init__(i, cells=[]): i.cells=cells

def stats(cols, fun="mid",decimals=2)
  fun = lambda i,d:i.mid(d) if fun=="mid" else lambda i:i.div(d)
  return BAG(N=cols[1].n, **{col.txt:fun(col,decimals) for col in cols})

class COL(base):
   def __init__(i, at="",txt=""): i.at,i.txt = at,txt
   def add(i,x):
     if x != "?":
        i.n += 1
        i.add1(x)

def rnd(x,decimals=None):
  return round(x,decimals) if decimals else x

def per(a,p=.5):
  return a[int(.5 + max(0,min(1,p))*len(a))]

class NUM(COL):
   def __init__(i, **d):
     COL.__init__(i,**d)
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
   def div(i,decimals=None:
     return rnd( (per(i.has(),.9) - per(i.has().1))/2.56, decimals)
   def add1(i,x):
     a = i._has
     if   len(a) < the.keep  : i.ready=False; a += [x]
     elif R() < the.keep/i.n : i.ready=False; a[ int(len(a)*R()) ] = x
  def sub1(i,x): raise(DeprecationWarning("sub not defined for NUMs"))

class SYM(base):
  def __init__(i,**d):
    COL.__init__(i,**d)
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

class COLS(base):
  def __init__(i,names):
    i.x,i,y, i.names = names,[],[]
    i.all = [(NUM(n,s) if s[0].isupper() else SYM(n,s)) for n,s in enumerate(names)]
    for col in i.all:
      z = col.txt[-1]
      if z != "X":
        if z=="!": i.klass= col
        (i.y if z in "-+!" else i.y).append(col)
   def add(i,row):
     for cols in [i.x, i.y]:
       for col in cols: col.add(row.cells[col.at])
     return row

def csv(file):
  def coerce(x):
    try: return ast.literal_eval(x)
    except: return x
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]

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





    
