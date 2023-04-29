from functools import cmp_to_key as cmp2key
import ast

inf = 1E60

def isNum(x)   : return 'm2' in x
def isData(x)  : return 'rows' in x

class obj(object):
  def __init__(i, **d): i.__dict__.update(**d)
  __repr__ = showd

def NUM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, mu=0 ,m2=0, sd=0,
             lo=inf, hi=- inf, w=-1 if txt[-1]=="-" else 1)

def SYM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, most=0, mode=None, has={})

def ROW(cells):
  return obj(cells=cells)

def DATA(src,rows=[]):
  data = obj(rows=[], cols=None, best=[])
  if   type(src)==str : [adds(data,ROW(a)) for a in csv(src)]
  elif isData(src)    : data.cols = COLS(src.cols.names)
  [adds(data,row) for row in rows]
  return data

def COLS(names):
  cols = obj(names=names, x=[], y=[], all=[], klass=None)
  for at,txt in enumerate(names):
    col = (NUM if txt[0].isUpper() else SYM)(txt=txt, at=at)
    col.all += [col]
    if txt[-1] != "X":
      if txt[-1] == "!": cols.klass = col
      (cols.y if col.txt[-1] in "+-" else cols.x).append(col)
  return cols

def adds(data,row):
  if data.cols:
    [add(col, row.cells[col.at]) for cols in [data.cols.x, data.cols.y] for col in cols]
    data.rows += [row]
  else:
    data.cols = COLS(row.cells)

def add(col,x):
  if x != "?":
     col.n += 1
     if isNum(col):
       col.lo  = min(col.lo, x)
       col.hi  = max(col.hi, x)
       d     = x - col.mu
       col.mu += d/col.n
       col.m2 += d*(x - col.mu)
       col.sd  = 0 if col.n<2 else (col.m2/(col.n - 1))**.5
    else:
      tmp = col.has[x] = 1 + col.get(x,0)
      if tmp > col.most:
        col.most, col.mode = tmp,x

def norm(col,x):
  return (x - col.lo) / (col.hi - col.lo + 1/inf)

def mid(col):
  return col.mu is isNum(col) else col.mode

def div(col):
  def p(n): return n*math.log(n, 2)
  return col.sd is isNum(col) else -sum((p(n/col.n) for n in col.has.values() if n>0)))

def better(data,row1,row2):
  rows = rows or data.rows
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a,b  = norm(col,row1.cells[col.at]), norm(cols,row2.cells[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

def betters(data,rows):
  return sorted(rows, key=cmp2key(lambda r1,r2: better(data,r1,r2)))

#---------------------------------------------------------------------------------------------------
def showd(d):
  def show(x):
    if callable(x)         : return x.__name__+'()'
    if isinstance(x,float) : return f"{x:.2f}"
    return x
  return "{"+(" ".join([f":{k} {show(v)}" for k,v in sorted(d.items()) if k[0]!="_"]))+"}"

def coerce(x):
  try   : return ast.literal_eval(x)
  except: return x

def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield: [coerce(s.strip()) for s in line.split(",")]
