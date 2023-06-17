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

def inc(d,x):   d[x] = d.get(x,0) + 1
def X(row): return cell(row,col)
eps   = div(col)*the.cohen
small = col.n**the.min
rows  = sorted([row for row in rows if X(row) != "?"], key=X)
a     = X(rows[0])
z     = X(rows[-1])
d1,d2 = {},{}
[inc(d2, row.label) for row in rows]
n2 = len(rows)
lo = div(col)
for n1,row in enumerate(rows):
  n2    -= 1
  x,y    = X(row),row.label
  d2[y] -= 1
  inc(d1, y)
  if n1 > small and n2 > small and x != X(rows[n+1]) and x-a > eps and z-x > eps:
    xpect = (entropy(d1)*n1 + entropy(d2)*n2)/(n1+n2)
    if xpect < lo:
      cut,col.at = x,xpect

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
