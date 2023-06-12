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
  -r  --rest    ratio best:rest       = 4
  -s  --seed    random number seed    = 1234567891
  -t  --top     explore top  ranges   = 8
  -w  --want    goal                  = "mitigate"
"""
import random,math,sys,ast,re
from termcolor import colored
from functools import cmp_to_key
from ast import literal_eval as thing
#---------------------------------------------
class BAG(dict): __getattr__ = dict.get
the = BAG(**{m[1]:thing(m[2])
             for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",__doc__)})
#---------------------------------------------
class base(object):
  def __repr__(i): 
    return i.__class__.__name__+str({k:v for k,v in i.__dict__.items() if k[0] != "_"})
#---------------------------------------------
class ROW(base):
  def __init__(i, cells=[]): i.cells,i.klass = cells,None
#---------------------------------------------
class COL(base):
  def __init__(i, txt="",at=0): i.n,i.at,i.txt = 0,at,txt
  def add(i,x):
    if x != "?":
      i.n += 1
      i.add1(x)
    return x
#---------------------------------------------
class NUM(COL):
  def __init__(i, txt="",at=0):
    COL.__init__(i,txt=txt,at=at)
    i.w = -1 if len(i.txt) > 0 and i.txt[-1] == "-" else 1
    i.mu = i.m2 = 0
    i.lo, i.hi = big, -big 
  def add1(i,x):
    i.lo = min(x, i.lo)
    i.hi = max(x, i.hi)
    delta = x - i.mu
    i.mu += delta/i.n
    i.m2 += delta*(x - i.mu)
  def div(i, decimals=None):
    return rnd((i.m2/(i.n - 1))**.5 if i.m2>0 and i.n > 1 else 0, decimals)
  def mid(i, decimals=None):
    return rnd(i.mu, decimals)
  def norm(i,x):
    return x if x=="?" else  (x-i.lo)/(i.hi - i.lo + 1/big)
#---------------------------------------------
class SYM(COL):
  def __init__(i,txt="",at=0):
    COL.__init__(i,txt=txt,at=at)
    i.counts,i.mode, i.most = {},None,0
  def add1(i,x):
    now = i.counts[x] = 1 + i.counts.get(x,0)
    if now > i.most: i.most, i.mode = now, x
  def div(i, decimals=None):
    a = i.counts
    return rnd( - sum(a[k]/i.n * math.log(a[k]/i.n,2) for k in a if a[k] > 0), decimals)
  def mid(i,decimals=None):
    return i.mode
  def sub(i,x):
    i.n -= 1
    i.counts[x] -= 1
    return x
#---------------------------------------------
class COLS(base):
  def __init__(i,names):
    i.x, i.y, i.names = [],[],names
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
  def sort(i,row1,row2):
    s1, s2, n = 0, 0, len(i.cols.y)
    for col in i.cols.y:
      a, b = col.norm(row1.cells[col.at]), col.norm(row2.cells[col.at])
      s1  -= math.exp(col.w * (a - b) / n)
      s2  -= math.exp(col.w * (b - a) / n)
    return s1 / n < s2 / n
  def sorts(i,rows=[]):
    return sorted(rows or i.rows, key=cmp_to_key(lambda a,b: i.sort(a,b)))
#---------------------------------------------
def tree(data):
  lst   = data.sorts()
  n     = int(len(data.rows)**the.min)
  bests = lst[-n:]
  rests = random.sample(lst[:-n], the.rest * n)
  for row in bests: row.klass = True
  for row in rests: row.klass = False
  all = bests + rests
  return tree1(data, all, len(all)**the.min)

def tree1(data,rows,stop, at=at,val=val,op=op,txt=txt):
  t = BAG(at=at,val=val,op=op,txt=txt,left=None,right=None,here=data.clone(rows))
  if len(rows) > 2*stop:
    _,at,op,val,txt = sorted((cut(data,c,rows) for c in data.cols.x))[0]
    left,right = [],[]
    [(left if t.op(row.cells[t.at], t.val) else right).append(row) for row in rows]
    if stop < len(left)  < len(rows):
      t.left  = tree1(data, left,  stop, at=at,val=val,txt=txt,op=op)
    if stop < len(right) < len(rows):
      t.right = tree1(data, right, stop, at=at,val=val,txt=txt,op=negate(op))
  return t

def showTree(t, lvl="",b4=""):
  if t:
    print(lvl + b4,str(len(t.here.rows)))
    pre= f"if {t.txt} {t.op.__doc__} {t.val}" if t.left or t.right else ""
    showTree(t.left,  lvl+"|.. ", pre)
    showTree(t.right, lvl+"|.. ", "else")

def negate(a):
  if a==fromFun: return toFun
  if a==toFun:   return fromFun
  if a==atFun:   return awayFun
  if a==awayFun: return toFun

def fromFun(x,y):
  ">"
  return x=="?" or y=="?" or x > y

def toFun(x,y):
  "<="
  return x=="?" or y=="?" or x <= y

def atFun(x,y):
  "=="
  return x=="?" or y=="?" or x == y

def awayFun(x,y):
  "!="
  return x=="?" or y=="?" or x == y

def cut(data,col,rows):
  return (cutNUM if isa(col,NUM) else cutSYM)(data,col,rows)

def cutSYM(_,col,rows):
  d = {}
  for row in rows:
    x1 = row.cells[col.at]
    if x1 != "?":
      if x1 not in d: d[x1] = SYM()
      d[x1].add(row.klass)
  return sorted((d[k].div(),col.at,atFun,k,col.txt) for k in d)[0]

def cutNUM(data,col,rows):
  lo = eps  = col.div()*the.cohen
  small     = len(rows)**the.min
  x         = lambda row: row.cells[col.at]
  y         = lambda row: row.klass
  xs,ys0,ys = NUM(), SYM(), SYM()
  rows      = sorted([row for row in rows if x(row) != "?"], key=x)
  cut       = x(rows[0])
  for row in rows: xs.add(x(row)); ys.add(y(row))
  for row in rows:
    ys0.add( ys.sub( y(row) ))
    if ys0.n > small and ys.n > small:
      if x(row) - x(rows[0]) > eps and x(rows[-1]) - x(row) > eps:
        xpect = (ys0.n*ys0.div() + ys.n*ys.div()) / (ys0.n+ys.n)
        if xpect < lo:
          cut,lo = x(row),xpect
  return lo,col.at,toFun,cut,col.txt
#---------------------------------------------
R   = random.random      # short cut to random number generator
isa = isinstance         # short cut for checking types
big = 1E30               # large number

def csv(file,filter=lambda x:x):
  def coerce(x):
    try: return thing(x)
    except: return x
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield filter([coerce(s.strip()) for s in line.split(",")])

def rnd(x,decimals=None):
  return round(x,decimals) if decimals else x

def rows(file): return csv(file, ROW)

def stats(cols, fun="mid", decimals=2):
  def what(col): return (col.mid if fun=="mid" else col.div)(decimals)
  return dict(mid=BAG(N=cols[1].n, **{col.txt:what(col) for col in cols}))
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

def sortsEg():
   d = DATA(rows(the.file))
   lst = d.sorts()
   m   = int(len(lst)**.5)
   best= d.clone(lst[-m:]); print("best",stats(best.cols.y))
   rest= d.clone(lst[:-m]);  print("rest",stats(rest.cols.y))

def treeEg():
  d = DATA(rows(the.file))
  showTree( tree(d) )

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
