#<!-- vim: set ts=2 sw=2 et: -->
from functools import cmp_to_key as cmp2key
import math,fire,sys,ast,re
inf = 1E60
#------------------------------
class obj(object):
  def __init__(self, **d): self.__dict__.update(**d)
  def __repr__(self):
    return "{"+(" ".join([f":{k} {show(v)}" for k,v in self.__dict__.items() if k[0]!="_"]))+"}"

def NUM(txt=" ",at=0):
  return obj(at=at, txt=txt, n=0, mu=0 ,m2=0, sd=0,
             lo=inf, hi=- inf, w=-1 if txt[-1]=="-" else 1)

def num1(num,x):
  if x == "?": return
  num.n  += 1
  num.lo  = min(num.lo, x)
  num.hi  = max(num.hi, x)
  d       = x - num.mu
  num.mu += d/num.n
  num.m2 += d*(x - num.mu)
  num.sd  = 0 if num.n<2 else (num.m2/(num.n - 1))**.5

def DATA(file):
  data = obj(rows=[], nums=[], y=[])
  for row in csv(file):
    if data.nums:
      [num1(num,row[num.at]) for num in data.nums]
      data.rows += [row]
    else:
      data.nums = [NUM(txt,at) for at,txt in enumerate(row) if txt[0].isupper()]
      data.y    = [col for col in data.nums if col.txt[-1] in "-+"]
  return data

def ordered(data):
  return sorted(data.rows, key=cmp2key(lambda r1,r2: better(data.nums,r1,r2)))

def better(nums,row1,row2):
  s1, s2,  n = 0, 0, len(nums)
  for col in nums:
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
def main(repeats=1, ja=23,file="../data/auto93.csv"):
  """simple rule generation"""
  data = DATA(file)
  print(data.y)
  #ordered(data)

fire.Fire(main)
