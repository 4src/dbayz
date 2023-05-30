# vim: set et sts=2 sw=2 ts=2 :
import re

class obj(object):
  oid = 0
  def __init__(i,**kw): obj.oid+=1; i.__dict__.update(_id=obj.oid, **kw)
  def __repr__(i)     : return i.__class__.__name__+printd(i.__dict__)
  def __hash__(i)     : return i._id

the = obj(file="../data/auto93.csv")

def HEADER(i,row):
  i.names = row
  for n,s in enumerate(row):
    if s[-1] in "-+!": i.goalp[n]=True
    if s[0].isupper():
      i.nums[n] = []
      i.lo[n], i.hi[n] = 10**30, -10**30
    else:
      i.counts[n] = {}

def ROW(i,row,filter):
  i.rows += [row]
  for n,x in enumerate(row):
    x = row[n] = filter(row[n])
    if x != "?":
      if n in i.nums:
        i.nums[n] += [x]
        i.lo[n]    = min(x, i.lo[n])
        i.hi[n]    = max(x, i.hi[n])
      else:
        i.counts[n][x] = 1 + i.counts[n].get(x,0)

def DATA(i=None,src=[],filter=lambda x:x):
  i = i or obj(names=None,rows=[],goalp={},nums={},lo={},hi={},counts={})
  for row in src:
    ROW(i,row,filter) if i.names else HEADER(i,row)
  i.nums = {k:sorted(v) for k,v in i.nums.items()}
  return i

def div(data,c):
  return stdev(data.nums[c]) if c in data.nums else entropy(data.counts[c])

def mid(data,c):
  return median(data.nums[c]) if c in data.nums else mode(data.counts[c])

def stats(data, cols=None, fun=mid):
  fun1 = lambda c: round(fun(data,c),2) if c in data.nums else fun(data,c)
  tmp = {c:fun1(c) for c in (cols or data.names)}
  return obj(N=len(data.rows),**tmp)
#---------------------------------------------------------------------------------------------------
def per(a,p=.5):
  n = max(0, min(len(a), int( p*len(a))))
  return a[n]

def median(a): return per(a)
def stdev(a) : return (per(a,.9) - per(a,.1))/2.56

def mode(d):
  hi = -1
  for k,v in d.items():
    if v > hi: hi,mode = v,k
  return mode

def ent(a):
  N = sum(a[k] for k in a)
  return - sum(a[k]/N * math.log(a[k]/N,2) for k in a if a[k] > 0)

def coerce(x):
  try: return int(x)
  except:
    try: return float(x)
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

