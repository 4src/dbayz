# vim: set et sts=2 sw=2 ts=2 :
"""
ragworm.py : the smallest AI brain I can imagine.    
(c) 2023 Tim Menzies <timm@ieee.og> BSD-2

USAGE:    
  python3 -B tests.py [OPTIONS] [-g ACTION]   

OPTIONS:  

  -b --bins   default number of bins                       = 16  
  -c --cohen  cohen's delta                                = .5  
  -f --file   data file                                    = ../data/auto93.csv  
  -g --go     start up action                              = nothing  
  -h --help   show help                                    = False  
  -k --k      Naive Bayes, low class frequency control     = 1  
  -m --m      Naive Bayes, low attribute frequency control = 2  
  -M --Min    recursion stops at N^M                       = .5  
  -r --rest   look at rest*|best| items                    = 3  
  -s --seed   random number seed                           = 1234567891   
  -S --Some   keep at least this number of numbers         = 256   
  -w --want   goal: plan,watch,xplore,doubt                = plan   
"""
from lib import *
the=settings(__doc__)
#------------------------------------------------ --------- --------- ----------
def SYM(c=0,s=" "):
  "Summarize stream of symbols."
  return BAG(ako=SYM, at=c, txt=s, n=0, has={},mode=None,most=0)

def NUM(c=0,s=" "):
  "Summarize stream of numbers."
  return BAG(ako=NUM, at=c, txt=s, n=0,
              mu=0,m2=0,
              lo=inf, hi=-inf, w = -1 if s[-1]=="-" else 1)

def COLS(words):
  """Factory for generating summary objects. Should be called  on the row
  columns names, top of a csv file. Upper case words become NUMs, others 
  are SYMs. Goals (ending in `+-!`) are added to a `y` list and others are 
  added to `x`. Anything ending in `X` is something to ignore."""
  cols = BAG(ako=COLS, names=words, x=[], y=[], all=[], klass=None)
  for c,s in enumerate(words):
    col = (NUM if s[0].isupper() else SYM)(c,s)
    cols.all += [col]
    if s[-1] != "X":
      if s[-1]=="!": klass=col
      (cols.y if s[-1] in "-+" else cols.x).append(col)
  return cols

def DATA(src, rows=[]):
  """Factory for making a `data` object either from a csv file (if `src` is a
  file name) or copying the structure of another `data` (f `src` is a `data`).
  Optionally, the new data can be augmented with `rows`."""
  data = BAG(ako=DATA, cols=[], rows=[])
  if type(src)==str   : [adds(data, ROW(a)) for a in csv(src)]
  elif src.ako is DATA: data.cols = COLS(src.cols.names)
  [adds(data,row) for row in rows]
  return data

def ROW(a):
  "Make a row containing `cells` to store data."
  return BAG(ako=ROW, cells=a, cooked=a[:])

def BIN(col,lo):
  """Create a `bin` for some column that stores rows. This is a place  to remember
  the labels seen in every row, and the `lo,hi` values seen in that column."""
  return BAG(ako=BIN, col=col, lo=lo, hi=lo, rows=[], ys=SYM())
#------------------------------------------------ --------- --------- ----------
def adds(data,row):
  "Summarize `row` inside `data` (and  keep `row` in `data.rows`)."
  if data.cols:
    data.rows += [row]
    for col in data.cols.all: add(col,row.cells[col.at])
  else:
    data.cols = COLS(row.cells)

def add(col,x,inc=1):
  "Increment counts of symbols seen (in SYMs), or numbers kept (in NUMs)."
  if x == "?": return x
  col.n += inc
  if col.ako is SYM:
    tmp = col.has[x] = col.has.get(x,0) + inc
    if tmp > col.most: col.most,col.mode = tmp,x
  else:
    col.lo = min(x, col.lo)
    col.hi = max(x, col.hi)
    d       = x - col.mu
    col.mu += d/col.n
    col.m2 += d*(x - col.mu)
  return x

def mid(col):
  "Return central tendency."
  return col.mode if col.ako is SYM else col.mu

def div(col):
  "Return diversity (tendency NOT to be at the central point)"
  return ent(col.has) if col.ako is SYM else (col.m2/(col.n - 1))**.5

def stats(data, cols=None, fun=mid):
  "Return a summary of `cols` in `data`, using `fun` (defaults to `mid`)."
  tmp = {col.txt: fun(col) for col in (cols or data.cols.y)}
  tmp["N"] = len(data.rows)
  return BAG(**tmp)

def norm(num,x):
  "Normalize `x` 0..1 for min..max."
  return x if x=="?" else (x - num.lo)/(num.hi - num.lo + 1/inf)

def constrasts(data1,data2):
  for row in data1.rows: row.y= True
  for row in data2.rows: row.y= False
  data12 = data(data1, data1.rows+ data2.rows)
  for col in data12.cols.x):
    tmp  = []
    here = lambda row:row.cells[col.at]
    rows = [row for row in data.rows if here(row) != "?"]
    for row in sorted(rows,key=here):
      x = here(row)
      k = (col.at, bin(col, x))
      if k not in tmp: tmp[k] = BIN(col1.at, x)
      tmp[k].rows += [row]
      tmp[k].lo = min(tmp[k].lo, x)
      tmp[k].hi = max(tmp[k].hi, x)
      add(tmp[k].y, row.y)
    tmp =sorted([bin for bin in tmp.values()], key=lambda b:b.lo)
    out += merges(col, tmp)

def bin(col,x):
  if col.ako is SYM: return x
  tmp = (col.hi - col.lo)/(the.bins - 1)
  return 1 if col.hi == col.lo else int(x/tmp + .5)*tmp

def merges(col,b4):
  if col.ako is SYM: return b4
  eps  = div(col)*the.cohen
  tiny = col.n/the.bins
  i,now = 0,[]
  while i < len(b4):
    a = b4[i]
    if i < len(b) - 1:
      b = b4[i+1]
      if c := merged(a, b, eps,tiny)
        a = c
        i += 1
    now += [a]
    i += 1
  return fillIntTheGaps(b4) if len(b4)==len(now) else merges(col,now)

def merged(a,b,eps,tiny):
  c    = deepcopy(a)
  c.lo = min(c.lo, b.lo)
  c.hi = max(c.hi, b.hi)
  for s,n in b.has.items(): add(c.y, s,n)
  if a.hi - a.lo < eps or b.hi - b.lo < eps: return c
  if a.n < tiny or b.n < tiny              : return c
  if div(c.y) <= xpect(a.y,b.y,div)        : return c

def fillInTheGaps(a):
  a[0].lo, a[-1].hi = -inf, inf
  for i in range(len(a)-1): a[i].hi = a[i+1].lo
  return a

def want(b,r,B,R):
  b,r = b/(B+1/inf),r/(R+1/inf)
  return dict(plan   = lambda : b**2/(b+r),
              watch  = lambda : r**2/(b+r),
              xplore = lambda : 1/(b+r),
              doubt  = lambda : (b+r)/abs(b - r))[the.want]

#------------------------------------------------ --------- --------- ----------
def better(data, row1, row2):
  "`Row1` is better than `row2` if moving to it losses less than otherwise."
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a, b = norm(col,row1.cells[col.at]), norm(col,row2.cells[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

def betters(data, rows=None):
  "Divide `data` into `best` and `rest`. Returns `best` and `rest` as `datas`."
  rows = sorted(rows or data.rows,
               key = cmp_to_key(lambda r1,r2:better(data,r1,r2)))
  cut = len(rows) - int(len(rows))**the.Min
  best,rest = [],[]
  for i,row in enumerate(rows):
    row.y = i > cut
    (best if i > cut else rest).append(row)
  return DATA(data,best), DATA(data,random.sample(rest, len(best)*the.rest))

