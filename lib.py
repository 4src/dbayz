# vim: set ts=2:sw=2:et:
import re
import ast
import sys
import math
import random
import traceback
from copy import deepcopy
from pyfiglet import Figlet
from termcolor import colored
from copy import deepcopy
from functools import cmp_to_key

seed = random.seed
r    = random.random

inf  = sys.maxsize / 2
ninf = -inf + 1
#------------------------------------------------ --------- --------- ----------
def figfont(txt,font):
  return Figlet(font=font).renderText(txt)

def yell(s,c):
  print(colored(s,"light_"+c,attrs=["bold"]),end="")

def showd(d): return "{"+(" ".join([f":{k} {show(v)}"
                         for k,v in d.items() if k[0]!="_"]))+"}"

def show(x):
  if callable(x)         : return x.__name__+'()'
  if isinstance(x,float) : return f"{x:.2f}"
  return x
#------------------------------------------------ --------- --------- ----------
class BAG(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  __repr__    = showd
#------------------------------------------------ --------- --------- ----------
def xpect(a,b,fun)
  return (a.n*fun(a) + b.n*fun(b))/(a.n+b.n)

def per(a, p=.5, key=lambda x:x):
  p=int(len(a)*p); p=max(0,min(len(a)-1,p)); return key(a[p])

def median(a, key=lambda x:x):
  return key(per(a,.5))

def stdev(a, key=lambda x:x):
  return (key(per(a,.9)) - key(per(a,.1)))/2.56

def ent(d):
  n = sum(( d[k] for k in d))
  return -sum((d[k]/n*math.log(d[k]/n,2) for k in d if d[k]>0))
#------------------------------------------------ --------- --------- ----------
def csv(file):
  with open(file) as fp:
    for line in fp:
      line = re.sub(r'([\n\t\r"\' ]|#.*)', '', line)
      if line:
        yield [coerce(cell.strip()) for cell in line.split(",")]

def coerce(x):
  if x=="?": return x
  try: return ast.literal_eval(x)
  except: return x

def settings(help, update=False):
  "Parses help string for lines with flags (on left) and defaults (on right)"
  d={}
  for m in re.finditer(r"\n\s*-\w+\s*--(\w+)[^=]*=\s*(\S+)",help):
    k,v = m[1], m[2]
    d[k] = coerce(v)
  d["_help"] = help
  return BAG(**d)

def cli(d):
  for k,v in d.items():
    v=str(v)
    for i,x in enumerate(sys.argv):
      if ("-"+k[0]) == x:
         v="False" if v=="True" else ("True" if v=="False" else sys.argv[i+1])
      d[k] = coerce(v)
  return d
#------------------------------------------------ --------- --------- ----------
def powerset(s):
  x = len(s)
  for i in range(1 << x):
     if tmp :=  [s[j] for j in range(x) if (i & (1 << j))]:
         yield tmp
#------------------------------------------------ --------- --------- ----------
def runs(the,funs):
  the=cli(the)
  if the.help:  return yell(the._help,"yellow")
  funs = [fun for fun in funs if re.match("^"+the.go, fun.__name__)]
  if len(funs) > 1:
    print(figfont("tests","ogre"),end="")
  n = sum([run(fun,the) for fun in funs])
  if len(funs) > 1:
    yell(f"{n} FAILURE(S)\n","red") if n>0 else yell("ALL PASSED\n","green")
  sys.exit(n)

def run(fun, settings):
  fail, cache = False, {k:settings[k] for k in settings}
  try:
    yell((fun.__name__ or "fun")+"\t","yellow")
    print(fun.__doc__ or ""," ",end="")
    seed(settings.seed)
    fail = fun() == False
  except:
    fail = True
    print(traceback.format_exc())
  yell("✘\n","red") if fail else yell("✔\n","green")
  for k in cache: settings[k] = cache[k]
  return fail
