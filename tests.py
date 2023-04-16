# vim: set ts=2:sw=2:et:
from dbayz import *

egs=[]
def eg(fun): global egs; egs += [fun]; return fun

@eg
def thed():
  "show options"
  print(str(the)[:30],"... ",end="")

@eg
def power():
  "powerset"
  print([x for x in powerset([1,2,3])],end=" ")
  return 2**3 -1 == len([x for x in powerset([1,2,3])])


@eg
def csvd():
  "read csv"
  return 3192==sum((len(a) for a in csv(the.file)))

@eg
def lohid():
  "find num ranges"
  num = NUM()
  [add(num,x) for x in range(20)]
  return 0==num.lo and 19==num.hi

@eg
def numd():
  "collect numeric stats"
  num = NUM()
  [add(num,r()) for x in range(10**4)]
  return .28 < div(num) < .32 and .46 < mid(num) < .54

@eg
def symd():
  "collect symbolic stats"
  sym = SYM()
  [add(sym,c) for c in "aaaabbc"]
  return 1.37 < div(sym) < 1.38 and mid(sym)=='a'

@eg
def statd():
  "collect stats from data"
  data0 = DATA("../data/auto93.csv")
  data1,data2 = betters(data0)
  s0 = stats(data0)
  s1 = stats(data1)
  s2 = stats(data2)
  a,l,m="Acc+", "Lbs-", "Mpg+"
  rnd=lambda z: round(z,ndigits=2)
  print([rnd(x) for x in [s2[m],s0[m],s1[m]]])
  print({c1.txt:rnd(div(c2)/(div(c1)+1/inf))
         for c1,c2 in zip(data1.cols.x, data2.cols.x)})
  return s1[a] > s2[a] and s1[m] > s2[m] and s1[l] < s2[l]

if __name__ == '__main__': runs(the,egs)
