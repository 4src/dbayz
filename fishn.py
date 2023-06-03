#!/usr/bin/env python3 -B
#<!--- vim: set et sts=2 sw=2 ts=2 : --->
#<img align=right width=200 src=fishn.png>
"""
## SYNOPSIS:
  fishn: look around just a little, guess where to search.

## USAGE:
  ./fishn.py [OPTIONS] [-g ACTIONS]
  
## DESCRIPTION:
  Use to find best regions within rows of data with multiple objectives.
  N rows of data are ranked via a multi-objective domination predicate
  and then discretized, favoring ranges that distinguish the best
  (N^min) items from a sample of the rest*(N^min)
  
## OPTIONS:
  
     -b  --bins    max number of bins    = 16
     -c  --cohen   size significant separation = .35
     -f  --file    data csv file         = ../data/auto93.csv
     -g  --go      start up action       = nothing
     -h  --help    show help             = False
     -k  --keep    how many nums to keep = 512
     -l  --lazy    laxy mode             = False
     -m  --min     min size              = .5
     -r  --rest    ratio best:rest       = 3
     -s  --seed    random number seed    = 1234567891
     -t  --top     explore top  ranges   = 16
     -w  --want    what goal to chase    = mitigate

(c) 2023, Tim Menzies, <timm@ieee.org>  BSD-2
"""
import random,math,sys,ast,re
from termcolor import colored
from functools import cmp_to_key
# ______
# Read in ROWs  into a DATA object. Find the
# best and rest ROWs. Find and sort the  BINs that discretize
# column values, favoring those that best select for what you want
# (defined at right).
# Return the best combination of the  `top` BINs. 
#       
# In `lazy` mode, try to do the above with fewest labels on the data.
# This can usually be achieved with only a handful of labels (5 to 20)
# but YMMV.
#  
# DATA can be built  from  memory or read from disk using the `csv` function. DATA
# has a first row listing column header names. Uppercase names are
# NUMeric (and others are SYMbolic). Names ending in "!,+,-" are goals
# to be achieved, maximized, minimized (respectively). Names ending in
# "X" denote columns we should ignore.
#  
# For examples on this code works, see the `egs` at end of files.
# Note one coding convention. UPPER CASE functions (or classes) are "factories"
# that generate objects of a particular type. If a factory name is used in lower
# case, that is an object from that factory; e.g. `data1` was made by DATA.
def want(b,r,B,R):
  "We have found `b` of the `B` best rows and 'r' or the `R` rest rows"
  b, r = b/(B + 1/inf), r/(R + 1/inf)
  match the.want:
    case "operate"  : return (b-r)        # want more b than r
    case "mitigate" : return b**2/(b+r)   # want lots of b and far less r
    case "monitor"  : return r**2/(b+r)   # want lots of r and far less b
    case "xtend"    : return 1/(b+r)      # want to go somewhere new
    case "xplore"   : return (b+r)/abs(b - r) # want the decision boundary
# ____
# ## Config

# `the` is where we store config options. These options are parsed
# out of the above doc string (using the `settings` function).
# Optionally, these options are updated from command-line (using the `cli` function).
the={} 

# ## Factories
# Factories make instances and have UPPER-CASE names (e.g. NUM, SYM, etc).
# In my code, any lower case factory names are instances (e.g. `num` is an instance of NUM).

# _________
# ## obj

# `obj` is  simple class that prints pretty, hashes easy, and inits easy.
class obj(object):
  oid = 0
  def __init__(i,**kw): obj.oid+=1; i.__dict__.update(_id=obj.oid, **kw)
  def __repr__(i)     : return printd(i.__dict__)
  def __hash__(i)     : return i._id

# ## ROW
def ROW(cells=[]):
  return obj(this=ROW,cells=cells)

# ## COLums
# NUM and SYM (which are "columns").
def COL(at,txt):
  return  (NUM if txt[0].isupper() else SYM)(at=at,txt=txt)

# All my columns count items seen (in `n`). Also, `col.this is NUM` is the idiom
# for recognizing a column of a particular type.

# ### SYM

# For example, SYMs summarizes streams of symbols (and SYMs knos frequency counts and `mode`).
def SYM(at=0,txt=""):
  return obj(this=SYM,txt=txt, at=at, n=0,
             counts={}, mode=None, most=0)

# NUMs summarizes streams of numbers (and NUMs know `lo,hi` and keeps a sample
# of those numbers in `_kept`). For goals, NUMs can have a weight of -1,1 denoting things to be
# minimized or maximized (respectively).
def NUM(at=0,txt=""):
   w = -1 if txt and txt[-1]=="-" else 1
   return obj(this=NUM,txt=txt, at=at, n=0,
              _kept=[], ok=True, w=w, lo=inf, hi=-inf)

# NUMs and SYMs can be incrementally updated.
def add(col,x,n=1):
  if x == "?": return
  col.n += n
  if col.this is SYM:
    now = col.counts[x] = 1 + col.counts.get(x,0)
    if now > col.most: col.most, col.mode = now, x
  else:
    col.lo = min(x, col.lo)
    col.hi = max(x, col.hi)
    a = col._kept
    if   len(a) < the.keep    : #  there is space, keep `x`, so just keep it
      col.ok=False; a += [x]
    elif r() < the.keep/col.n : # else, keep some things (by replace old things)
      col.ok=False; a[int(len(a)*r())] = x

# NUM _kept can get jumbled up so before we use it, make sure it is `ok` 
def ok(col):
  if col.this is NUM and not col.ok:
    col._kept.sort()
    col.ok=True
  return col

# `norm`alize a NUM value 0..1
def norm(num,x):
  return x if x=="?" else (x - num.lo) / (num.hi - num.lo + 1/inf)

# `mid` of a col is the central tendency and is mode or median for SYMs and NUMs, respectively.
def mid(col,decimals=None):
  return col.mode if col.this is SYM else rnd(median(ok(col)._kept),decimals)

# `div` of a col is the diversity (how far we stray from `mid`) and is  entropy or standard deviation for SYMs and NUMs, respectively.
def div(col,decimals=None):
  return rnd(ent(col.counts) if col.this is SYM else stdev(ok(col)._kept),decimals)

# `stats` returns an `obj` with `mid` or `div` on many columns. 
def stats(data, cols=None, fun=mid, decimals=2):
  return obj(N=len(data.rows),**{c.txt:fun(c,decimals) for c in (cols or data.cols.y)})

# ## COLS (makes many columns)

# Convert a list of strings to NUMs or SYMs. Anything ending with "X" is ignore.
# Upper case names become NUMs (and everything else is a SYM). For convenience, list
# all the independent/dependent variables together in `x.y` respectively.
def COLS(names):
  cols = obj(this=COLS, names=names, x=[], y=[], 
             all = [COL(n,s) for n,s in enumerate(names)])
  for col in cols.all:
    if col.txt[-1] != "X": 
      (cols.y if col.txt[-1] in "-+!" else cols.x).append(col)
  return cols

# ## DATA

# `DATA` instances store ROWs, and summarized those into columns.
def DATA(data=None, src=[]):
  data = data or obj(this=DATA,rows=[], cols=None)
  for row in src:
    if not data.cols: # reading row1 (list of column names)
      data.cols = COLS(row)
    else:
      row = ROW(row) if isinstance(row,list) else row # ensure we are reading ROWs
      data.rows += [row]
      for cols in [data.cols.x, data.cols.y]:
        for col in cols:
          add(col,row.cells[col.at])
  return data

# **DATA functions:**   
#  Copy the structure of a `data` table; 
def clone(data, rows=[]): return DATA(DATA(src=[data.cols.names]), rows)

# Sort `rows` worst to best using the `better` function
def betters(data,rows=[]):
  return sorted(rows or data.rows,
                key=cmp_to_key(lambda r1,r2: better(data,r1,r2)))

# `Row1` is better than `row2` if moving to it losses less than otherwise.
def better(data, row1, row2):
  s1, s2, cols, n = 0, 0, data.cols.y, len(data.cols.y)
  for col in cols:
    a, b = norm(col,row1.cells[col.at]), norm(col,row2.cells[col.at])
    s1  -= math.exp(col.w * (a - b) / n)
    s2  -= math.exp(col.w * (b - a) / n)
  return s1 / n < s2 / n
#---------------------------------------------------------------------------------------------------
# ## Discertization

# Map `x` to a small number of values. For SYMs, we just return the symbol. For NUMs,
# place into one of `the.bins` buckets.
def discretize(col,x):
  if x == "?": return
  if col.this is NUM:
    x = int(the.bins*(x - col.lo)/(col.hi - col.lo + 1/inf))
    x = min(the.bins, max(0, x))
  return x

# Track what `x` value ranges hold what `rows` (and those `rows`
# are divided inside `ys` according to their class.
class BIN(object):
  def __init__(self,at=0,txt=" ",lo=None,hi=None,B=0,R=0):
    self.at=at; self.txt=txt; self.lo=lo or inf; self.hi=hi or lo
    self.n=0; self.ys={}; self.score=0; self.B=B; self.R=R
  def __repr__(self):
    if self.hi == self.lo: return f"{self.txt}={self.lo}"
    if self.hi == inf:     return f"{self.txt}>={self.lo}"
    if self.lo == -inf:    return f"{self.txt}<{self.hi}"
    return f"{self.lo} <= {self.txt} < {self.hi}"

# Update the `x` ranges and the set of `rows` help in that `x` range.
def binAdd(bin, x, y, row):
  bin.n     += 1
  bin.lo     = min(bin.lo, x)
  bin.hi     = max(bin.hi, x)
  bin.ys[y]  = bin.ys.get(y,set())
  bin.ys[y].add(row)
  return bin

# Combine two `bins`.
def merge(bin1, bin2):
  out = BIN(at=bin1.at, txt=bin1.txt,
            lo=bin1.lo, hi=bin2.hi,
            B=bin1.B, R=bin2.R)
  out.n = bin1.n + bin2.n
  for d in [bin1.ys, bin2.ys]:
    for klass in d:
      old = out.ys[klass]  = out.ys.get(klass,set())
      out.ys[klass]  = old | d[klass]
  return out

# Return the merge of two bins, but only if that merge is useful.
def merged(bin1,bin2,num,best):
  out   = merge(bin1,bin2)
  eps   = div(num)*the.cohen
  small = num.n / the.bins
  if len(out.ys.get(best,[]))/out.B < 0.05: return out
  if bin1.n <= small or bin1.hi - bin1.lo < eps : return out
  if bin2.n <= small or bin2.hi - bin2.lo < eps : return out
  e1, e2, e3 = binEnt(bin1), binEnt(bin2), binEnt(out)
  if e3 <= (bin1.n*e1 + bin2.n*e2)/out.n : return out

# Returns the entropy of the row distributions in a `bin`.
def binEnt(bin):
  return ent({k:len(set1) for k,set1 in bin.ys.items()})

# ____
# ## Rule Generation

# Find attribute ranges that distinguish between the rows in
# `data1` and `data2`. If `elite` then sort the ranges by their
# `score`, then return the first `the.top` items.
def contrasts(data1,data2, elite=False):
  top,bins = None, _contrasts(data1,data2)
  n=0
  if elite:
    for bin in sorted(bins,reverse=True,key=lambda bin: bin.score):
      top = top or bin
      if not(bin.lo == -inf and bin.hi == inf) and bin.score > top.score*.1:
        n += 1
        if n <= the.top: yield bin
  else:
    for bin in bins: yield bin

# Divide values from each `data` into a few values, then
# merge uninformative divisions).
def _contrasts(data1,data2):
  data12 = clone(data1, data1.rows + data2.rows)
  for col in data12.cols.x:
    bins = {}
    for klass,rows in dict(best=data1.rows, rest=data2.rows).items():
      for row in rows:
        x = row.cells[col.at]
        if z := discretize(col, x):
          if z not in bins: bins[z] = BIN(at=col.at,txt=col.txt,lo=x,
                                          B=len(data1.rows), R=len(data2.rows))
          binAdd(bins[z], x, klass, row)
    for bin in merges(col, sorted(bins.values(), key=lambda z:z.lo),"best"):
      yield value(bin, col)

# Find ranges, expand the ranges to cover gaps in the data.
def merges(col,bins,best):
  if col.this is SYM: return bins
  bins = mergeds(bins,col,best)
  for j in range(len(bins)-1): bins[j].hi = bins[j+1].lo
  bins[0].lo = -inf
  bins[-1].hi =  inf
  return bins

# While there exists adjacent ranges that can be merged, merge them
# then look for other possible merges.
def mergeds(a, col,best):
  b,j = [],0
  while j < len(a):
    now = a[j]
    if j < len(a) - 1:
      if new := merged(a[j], a[j+1], col,best): now,j = new,j+1
    b += [now]
    j += 1
  return a if len(a) == len(b) else mergeds(b, col,best)

# Score a bin, on a range of possible criteria. 
def value(bin,col):
  b,r = bin.ys.get("best",set()), bin.ys.get("rest",set())
  bin.score = want(len(b),len(r), bin.B, bin.R)
  return bin

# Return  a row if its selected by a `bin`.
def select(bin,row):
  x = row.cells[bin.at]
  if x=="?"                         : return row
  if bin.lo == bin.hi == x          : return row
  if bin.lo == -inf and x <  bin.hi : return row
  if bin.hi ==  inf and x >= bin.lo : return row
  if bin.lo <= x and x < bin.hi     : return row

# Return  the rows selected by a set of bins.
def selects(bins,rows):
  d={}
  for bin in bins:
    here = d[bin.at] = d.get(bin.at, set())
    d[bin.at] = here | set([row for row in rows if select(bin,row)])
  out= None
  for set1 in d.values():
    if out: out = out & set1
    else  : out = set1
    if len(out)==0: return 
  if len(out) == len(rows): return
  return out

#---------------------------------------------------------------------------------------------------
# Short cuts
r=random.random
inf=float("inf")

# Print colors.
def red(s): return colored(s,"red",attrs=["bold"])
def green(s): return colored(s,"green",attrs=["bold"])
def yellow(s): return colored(s,"yellow",attrs=["bold"])
def bold(s): return colored(s,"white",attrs=["bold"])

# Round numbers, if `decimals` is set.
def rnd(x,decimals=None):
  return round(x,decimals) if decimals else  x

# Return an item at some step along its length.
def per(a,p=.5):
  n = max(0, min(len(a)-1, int( 0.5 + p*len(a))))
  return a[n]

# Return middle point of a list
def median(a): return per(a,.5)

# Plus or minus 1.28 standard deviations covers 90% of the data.
# Therefore, sd = (.9sd - .1sd)/2.56
def stdev(a) : return (per(a,.9) - per(a,.1))/2.56

# Returns the diversity of a set of symbols.
def ent(a):
  N = sum((a[k] for k in a))
  return - sum(a[k]/N * math.log(a[k]/N,2) for k in a if a[k] > 0)

# Returns all subsets.
def powerset(s):
  r = [[]]
  for e in s: r += [x+[e] for x in r]
  return r[1:]

# Converts strings to things.
def coerce(x):
  try: return ast.literal_eval(x)
  except: return x

# Pretty-print of a dictionary.
def printd(d):
  p= lambda x: '()' if callable(x) else (f"{x:.2f}" if isinstance(x,float) else str(x))
  return "{"+(" ".join([f":{k} {p(v)}" for k,v in d.items() if k[0]!="_"]))+"}"

# Read a csv file
def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(s.strip()) for s in line.split(",")]

# Parse the doc string (at top of file) to extract the settings.
def settings(s):
  setting = r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)"
  return obj(**{m[1]:coerce(m[2]) for m in re.finditer(setting,s)})
#---------------------------------------------------------------------------------------------------
# Test Suite Stuff

# Top-level control. Prints help or the number of `egs` that return `False`.
def runs():
  if the.help:
    headings, flags ="\n[#]+ [A-Z][A-Z]+:"," [-][-]?[\S]+"
    show  = lambda f: lambda m: f(m[0])
    print(re.sub(headings, show(yellow),re.sub(flags, show(bold), __doc__)))
  else:
    n= sum([run(s,fun) for s,fun in egs.items()
                       if s[0]!="_" and the.go in ["all",s]])
    print(red(f"{n} FAILURE(S)") if n>0 else green(f"{n} FAILURE(S)"))
    return n

# Run one example, initialize the random seed before and all the
# options afterwards
def run(s,fun):
  d = the.__dict__
  saved = {k:d[k] for k in d}
  random.seed(the.seed)
  print(bold(s) + " ",end="")
  out = fun()
  for k in saved: d[k] = saved[k]
  print(red("FAIL") if out==False else green("PASS"))
  return out==False

# Update a dictionary from command-line flagss
def cli(d):
  d1=d.__dict__
  for k,v in d1.items():
    v = str(v)
    for i,x in enumerate(sys.argv):
      if ("-"+k[0]) == x or ("--"+k) == x:
        d1[k]= coerce("True" if v=="False" else ("False" if v=="True" else sys.argv[i+1]))
  return d
#---------------------------------------------------------------------------------------------------

# Define examples.
egs={}
def eg(f): egs[f.__name__]= f; return f

@eg
def thed(): print(str(the),"...",end=" ")

@eg
def powered(): print(powerset("abc"),end=" ")

@eg
def colnum():
  num = NUM()
  [add(num, random.gauss(10,2)) for _ in range(the.keep)]
  return 9.95 <= mid(num) <= 10.05 and 1.9 <= div(num) <=2.1

@eg
def colnum2():
  num = NUM()
  the.keep=20
  [add(num, x) for x in range(1000)]
  print(ok(num)._kept)

@eg
def colnum3():
  return the.keep == 512

@eg
def colsym():
  sym = SYM()
  [add(sym,c) for c in "aaaabbc"]
  return 1.37 < div(sym) < 1.38 and mid(sym)=='a'

@eg
def csved(): return 3192==sum((len(a) for a in csv(the.file)))

@eg
def statd():
  data=DATA(src=csv("../data/auto93.csv"))
  print(stats(data,cols=data.cols.x))
  print(stats(data))
  print(stats(data,fun=div))

@eg
def orderer():
  data = DATA(src=csv(the.file))
  rows = betters(data)
  print("")
  print(data.cols.names)
  print("all ", stats(data))
  print("best", stats(clone(data,rows[-30:])))
  print("rest", stats(clone(data,rows[:30])))

@eg
def contrasted():
  data = DATA(src=csv(the.file))
  rows = betters(data)
  b = int(len(data.rows)**the.min)
  best = clone(data,rows[-b:])
  rest = clone(data,rows[:b*the.rest])
  print("\nall ", stats(data))
  print("best", stats(best))
  print("rest", stats(rest))
  b4 = None
  for bin in contrasts(best,rest):
    if bin.txt != b4: print("")
    print(f"{bin.txt:10} {bin.lo:5} {bin.hi:5}",
          obj(best=len(bin.ys.get("best",set())),
              rest=len(bin.ys.get("rest",set())),
              score=f"{bin.score:.2f}"))
    b4 = bin.txt

@eg
def bested():
  data = DATA(src=csv(the.file))
  rows = betters(data)
  b = int(len(data.rows)**the.min)
  best = clone(data,rows[-b:])
  rest = clone(data,rows[:b*the.rest])
  print("")
  for bin in contrasts(best,rest, elite=True):
    print(bin,bin.score)

@eg
def bested2():
  data = DATA(src=csv(the.file))
  print("\nALL:", stats(data))
  rows = betters(data)
  b = int(len(data.rows)**the.min)
  best = clone(data,rows[-b:])
  rest = clone(data,random.sample(rows,b*the.rest))
  print("")
  for bins in powerset(list(contrasts(best,rest, elite=True))):
    print(stats(clone(data, selects(bins, data.rows))),bins)

#---------------------------------------------------------------------------------------------------
# ## Start-up

the = settings(__doc__)
if __name__ == "__main__":
  the = cli(the)
  sys.exit(runs())
