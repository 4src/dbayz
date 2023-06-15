import random,re
from ast import literal_eval
from fileinput import FileInput as file_or_stdin
from functools import cmp_to_key

class BAG(dict): __getattr__ = dict.get
the= BAG(nums=256, bins=16, seed=1234567891,file="../data/auto93.csv")

INF=1E30
R=random.random
random.seed(the.seed)
#----------------------------------------------------------------
def NUM(at=0,txt=""):
  w= 0 if txt and txt[-1]=="-" else 1
  return BAG(ako=NUM,at=at,txt=txt,n=0,w=w,has=[],ok=False)

def SYM(at=0,txt=""):
  return BAG(ako=SYM,at=at,txt=txt,n=0, has={})

def COLS(names):
  all=[(NUM if s[0].isupper() else SYM)(at,s) for at,s in enumerate(names)]
  x,y=[],[]
  for col in all:
    if col.txt[-1] != "X":
      (y if col.txt[-1] in "+-" else x).append(col)
  return BAG(ako=COLS,all=all,y=y,x=x,names=names)

def DATA(data=None,rows=[]):
  data = data or BAG(rows=[],cols=None)
  [adds(data,lst) for lst in rows]
  return data
#----------------------------------------------------------------
def clone(data, rows=[]):
  return DATA( DATA(rows=[data.cols.names]), rows)

def adds(data,lst):
  if data.cols:
    data.rows += [lst]
    for cols in [data.cols.x, data.cols.y]:
      for col in cols: add(col, lst[col.at])
  else:
    data.cols = COLS(lst)

def add(col,x):
  def num():
    a = col.has
    if   len(a) < the.nums   : col.ok=False; a += [x]
    elif R() < the.nums/col.n: col.ok=False; a[int(len(a)*R())] = x
  def sym():
    col.has[x] = col.has.get(x,0) + 1
  if x!= "?":
    col.n += 1
    num() if col.ako is NUM else sym()

def ok(col):
  if col.ako is NUM and not col.ok: col.has.sort(); col.ok=True
  return col

def lo(num): return ok(num).has[0]
def hi(num): return ok(num).has[-1]

def mid(col):
  return mode(col.has) if col.ako is SYM else median(ok(col).has)

def div(col):
  return ent(col.has) if col.ako is SYM else stdev(ok(col).has)

def norm(col,x):
  if x=="?" or col.ako is SYM: return x
  return (x - lo(col))/(hi(col) - lo(col) + 1/INF)

def bins(col):
  if col.ako is SYM: return col.has
  d=SYM()
  a=ok(col).has
  for n in a:
    n= 1+int(norm(col,n)*the.bins)
    add(d, (a[0] + (a[-1] - a[0])/n))
  return d.has

def height(data,row):
  return (sum(abs(col.w - norm(col,row[col.at]))**2 for col in data.cols.y)
          / len(data.cols.y)
         )**.5

def sorter(data):
    return cmp_to_key(lambda a,b:  height(data,a) < height(data,b))

def stats(data, cols=None, fun=mid):
  return BAG(N=len(data.rows),**{col.txt: fun(col) for col in (cols or data.cols.y)})
#----------------------------------------------------------------
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
        yield [coerce(s.strip()) for s in line.split(",")]
#----------------------------------------------------------------

d=DATA(rows=csv(the.file))
d.rows.sort(key=sorter(d))
print(stats(d))
print(stats(clone(d, d.rows[:30])))
print(stats(clone(d, d.rows[-30:])))
for k,v in bins(d.cols.x[0]).items(): print(d.cols.x[0].txt,k,v)
