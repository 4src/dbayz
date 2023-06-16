import random,re
from ast import literal_eval
from functools import cmp_to_key
from fileinput import FileInput as file_or_stdin

class BAG(dict): __getattr__ = dict.get
the= BAG(nums=256, bins=8, seed=1234567891, file="../data/auto93.csv")

INF=1E30
R=random.random
random.seed(the.seed)
#--------------------
def NUM(at=0,txt=""):
  w= 0 if txt and txt[-1]=="-" else 1
  return BAG(this=NUM,at=at,txt=txt,n=0,w=w,has=[],ok=False)

def SYM(at=0,txt=""):
  return BAG(this=SYM,at=at,txt=txt,n=0, has={})

def COLS(names):
  all = [(NUM if s[0].isupper() else SYM)(at,s) for at,s in enumerate(names)]
  x,y = [],[]
  for col in all:
    if col.txt[-1] != "X":
      (y if col.txt[-1] in "+-" else x).append(col)
  return BAG(this=COLS,all=all,y=y,x=x,names=names)

def ROW(cells=[]):
  return BAG(this=ROW, cells=cells, label=None)

def DATA(data=None,rows=[]):
  data = data or BAG(rows=[],cols=None)
  [adds(data,row) for row in rows]
  return data
#------------------------
def clone(data, rows=[]):
  return DATA( DATA(rows=[ROW(data.cols.names)]), rows)

def cell(row,col): return row.cells[col.at]

def adds(data,row):
  if data.cols:
    data.rows += [row]
    for cols in [data.cols.x, data.cols.y]:
      for col in cols: add(col, cell(row,col))
  else:
    data.cols = COLS(row.cells)

def add(col,x):
  def sym(): col.has[x] = col.has.get(x,0) + 1
  def num():
    a = col.has
    if   len(a) < the.nums   : col.ok=False; a += [x]
    elif R() < the.nums/col.n: col.ok=False; a[int(len(a)*R())] = x
  if x!= "?":
    col.n += 1
    num() if col.this is NUM else sym()

def ok(col):
  if col.this is NUM and not col.ok: col.has.sort(); col.ok=True
  return col

def lo(num):  return ok(num).has[0]
def hi(num):  return ok(num).has[-1]
def mid(col): return mode(col.has) if col.this is SYM else median(ok(col).has)
def div(col): return ent(col.has)  if col.this is SYM else stdev(ok(col).has)

def norm(col,x):
  if x=="?" or col.this is SYM: return x
  return (x - lo(col))/(hi(col) - lo(col) + 1/INF)

def bin(col,x):
  if x !="?":
    if col.this is SYM: return x
    n = 1 + int(norm(col,x)*the.bins)
    a = ok(col).has
    return a[0] + (a[-1] - a[0])/n

def bins(col,best,rest):
  d={}
  for y,rows in dict(best=rest,rest=rest).items():
    for row in rows:
      x = cell(row,col)
      if k := bin(col,x):
        xy = d.get(k,None) or BAG(xs=NUM(col.at,col.txt), ys=SYM())
        add(xy.xs, x)
        add(xy.ys, y)
  return sorted(d.values(), key=lambda xy: lo(xy.xs))

#XXX must redo merge
def merge(bin1,bin2):
  out = BAG(lo=bin1.lo, hi=bin2.hi, y=SYM(bin1.y.at, bin1.y.txt))
  for sym in [bin1.y, bin2.y]:
    out.y.n += sym.n
    for k,v in sym.has.items(): sym.has[k] = sym.has.get(k,0) + v
  return out

def merges(bins):
  out = [bins[0]]
  for bin in bins[1:]: out += [merge(out[-1], bin)]
  return out

def height(data,row):
  return ( sum(abs(col.w - norm(col,cell(row,col)))**2 for col in data.cols.y)
           / len(data.cols.y) )**.5

def sorter(data):
    return cmp_to_key(lambda a,b:  height(data,a) < height(data,b))

def stats(data, cols=None, fun=mid):
  return BAG(N=len(data.rows),**{col.txt: fun(col) for col in (cols or data.cols.y)})
#----------------
def per(a, p=.5):
  return a[ max(0,min(len(a)-1, int(len(a)*p))) ]

def median(a): return per(a,.5)

def stdev(a): return (per(a,.9) - per(a,.1))/2.56

def ent(d):
  n = sum(( d[k] for k in d))
  return -sum((d[k]/n*math.log(d[k]/n,2) for k in d if d[k]>0))

def mode(d):
  max,out = 0,None
  for k,v in d.items():
    if v > max: out,max = k,v
  return out

def csv(file):
  def coerce(x):
    try: return literal_eval(x)
    except: return x
  with file_or_stdin(file) as src:
    for line in src:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield ROW([coerce(s.strip()) for s in line.split(",")])
#-------------------------
d=DATA(rows=csv(the.file))
d.rows.sort(key=sorter(d))
print(stats(d))
print(stats(clone(d, d.rows[:30])))
print(stats(clone(d, d.rows[-30:])))
#for k,v in bins(d.cols.x[0]).items(): print(d.cols.x[0].txt,k,v)
