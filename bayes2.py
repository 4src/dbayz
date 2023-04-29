#<!-- vim: set ts=2 sw=2 et: -->
from functools import cmp_to_key as cmp2key
import math,fire,sys,ast,re
inf = 1E60
#------------------------------
class obj(object):
  def __init__(self, **d): self.__dict__.update(**d)
  def __repr__(self):
    return "{"+(" ".join([f":{k} {show(v)}" for k,v in self.__dict__.items() if k[0]!="_"]))+"}"

def COL(txt=" ",  at=0, data=None):
   col = (NUM if txt[0].isupper() else SYM)(txt=txt,at=at)
   if data and txt[-1] != "X":
     (data.y if txt[-1] in "-+" else data.x).append(col)
   return col

def SYM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, has={}, mode=None,most=0,isNum=False)

def NUM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, mu=0 ,m2=0, sd=0,isNum=True,
             lo=inf, hi=- inf, w=-1 if txt[-1]=="-" else 1)

def clone(data,rows=[]):
  return adds(rows, adds([data1.names]))

def DATA(src,data=None):
  data = data or obj(rows=[],  names=[], all=[], x=[], y=[])
  for row in src:
    if data.all:
      [add(col,row[col.at]) for col in data.all]
      data.rows += [row]
    else:
      data.names= row
      data.all = [COL(txt,at,data) for at,txt in enumerate(row)]
  return data

def add(col,x):
  if x == "?": return
  col.n  += 1
  if col.isNum:
    col.lo  = min(col.lo, x)
    col.hi  = max(col.hi, x)
    d       = x - col.mu
    col.mu += d/col.n
    col.m2 += d*(x - col.mu)
    col.sd  = 0 if col.n<2 else (col.m2/(col.n - 1))**.5
  else:
    tmp = col.has[x] = 1 + col.has.get(x,0)
    if tmp >  col.most: col.most, col.mode = tmp,x

def ordered(data,rows=[]):
  return sorted(rows or data.rows,
                key=cmp2key(lambda r1,r2: better(data,r1,r2)))

def better(data,row1,row2):
  s1, s2,  n = 0, 0, len(data.y)
  for col in data.y:
    a,b  = norm(col,row1[col.at]), norm(col,row2[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n

def norm(col,x):
  return (x - col.lo) / (col.hi - col.lo + 1/inf)

#---------------------------------------------------------------------------------------------------
def show(x):
  if callable(x)         : return x.__name__+'()'
  if isinstance(x,float) : return f"{x:.2f}"
  return x

def coerce(x):
  try: return ast.literal_eval(x)
  except: return x

def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]
#---------------------------------------------------------------------------------------------------
def main(Repeats=1, ja=23,file="../data/auto93.csv"):
  """Simple rule generation"""
  data = DATA(csv(file))
  print(data.y)
  ordered(data)

if __name__ == "__main__": fire.Fire(main)
