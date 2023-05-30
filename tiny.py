# vim: set et sts=2 sw=2 ts=2 :
import random,re,ast

class obj(object):
  oid = 0
  def __init__(i,**kw): obj.oid+=1; i.__dict__.update(_id=obj.oid, **kw)
  def __repr__(i)     : return i.__class__.__name__+printd(i.__dict__)
  def __hash__(i)     : return i._id

the = obj(keep=512,file="../data/auto93.csv")

def SYM(at=0,txt=""): return obj(txt=txt, at=at, counts={},isNum=False)

def NUM(at=0,txt=""):
   w = -1 if txt and txt[-1]=="-" else 1
   return obj(txt=txt, at=at, _all=[], ok=True, w=w, lo=10**30, hi=-10**30,isNum=True)

def DATA(data=None, src=[], filter=lambda x:x):
  def header(cols,row):
    cols.names = row
    for n,s in enumerate(row):
      col = (NUM if s[0].isupper() else SYM)(at=n,txt=s)
      cols._all += [col]
      if s[-1] != "X":
        (cols.y if s[-1] in "-+!" else cols.x).append(col)
  def newRow(row,filter):
    data.rows += [row]
    for cols in [data.cols.x, data.cols.y]:
      for col in cols:
        x = row[col.at] = filter(row[col.at])
        if x != "?":
          if col.isNum:
            col.lo = min(x, col.lo)
            col.hi = max(x, col.hi)
            a = col._all
            if   len(a) < the.keep    : col.ok=False; a += [x]
            elif r() < the.keep/col.n : col.ok=False; a[int(len(a)*r())] = x
          else:
            col.counts[x] = 1 + col.counts.get(x,0)
  #------------------------------------------------
  data = data or obj(rows=[], cols=obj(names=None, x={}, y={}, _all=[]))
  for row in src:
    newRow(row,filter) if data.cols.names else header(data.cols,row)
  return data

def ok(col):
  if col.isNum and not col.ok:
    col._all.sort()
    col.ok=True
  return col

def norm(col,x):
  return x if x=="?" else (x-col.lo) / (col.hi - col.lo + 10**-30)

def div(col,decimals=None):
  return rnd(stdev(ok(col)._all) if col.isNum else entropy(col.counts), decimals)

def mid(col,decimals=None):
  return rnd(median(ok(col)._all),decimals) if col.isNum else mode(col.counts)

def stats(data, cols=None, fun=mid, decimals=2):
  tmp = {col.txt:fun(col,decimals) for col in (cols or data.cols.y).values()}
  return obj(N=len(data.rows),**tmp)

def better(data, row1, row2):
  "`Row1` is better than `row2` if moving to it losses less than otherwise."
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a, b = norm(col,row1[col.at]), norm(col,row2[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

#---------------------------------------------------------------------------------------------------
def settings(help, update=False):
  "Parses help string for lines with flags (on left) and defaults (on right)"
  d={}
  for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",help):
    k,v = m[1], m[2]
    d[k] = coerce(v)
  d["_help"] = help
  return BAG(**d)

r=random.random

def rnd(x,decimals=None):
   return round(x,decimals) if decimals else  x

def per(a,p=.5):
  n = max(0, min(len(a)-1, int( p*len(a))))
  return a[n]

def median(a): return per(a)
def stdev(a) : return (per(a,.9) - per(a,.1))/2.56

def mode(d):
  hi = -1
  for k,v in d.items():
    if v > hi: mode,hi = k,v
  return mode

def ent(a):
  N = sum(a[k] for k in a)
  return - sum(a[k]/N * math.log(a[k]/N,2) for k in a if a[k] > 0)

def coerce(x):
  try: return ast.literal_eval(x)
  except: return x

def printd(d):
  p= lambda x: '()' if callable(x) else (f"{x:.2f}" if isinstance(x,float) else str(x))
  return "{"+(" ".join([f":{k} {p(v)}" for k,v in d.items() if k[0]!="_"]))+"}"

def cli(d):
  for k,v in d.items():
    v= str(v)
    for i,x in enumerate(sys.argv):
      if ("-"+first(k)) == x or ("--"+k) == x:
        v= "False" if v=="True" else ("True" if v=="False" else sys.argv[i+1])
    d[k] = coerce(v)
  return d

def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]
#---------------------------------------------------------------------------------------------------
d=DATA(src=csv(the.file), filter=coerce)
print(stats(d))
