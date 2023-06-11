---
title: "How to Code Less"
subtitle: "(101 tips and tricks for smarter, smaller, coding)"
author: "Tim Menzies (timm@ieee.org)"
email: "timm@ieee.org"
date: 2023-06-17
linestretch: .85
documentclass: extreport
fontsize: 9pt
pagestyle: headings
classoption:
- twocolumn
- portrait
geometry: margin=15mm
header-includes:
    - \usepackage{titlesec}
    - \titleformat*{\section}{\itshape}
    - \setlength{\columnsep}{5cm}
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[CO,CE]{This is fancy}
    - \fancyfoot[CO,CE]{So is this}
    - \fancyfoot[LE,RO]{\thepage}
    - \RequirePackage[sc]{mathpazo}          % Palatino for roman (\rm)
    - \RequirePackage[scaled=0.95]{helvet}   % Helvetica for sans serif (\sf)
    - \RequirePackage{courier}               % Courier for typewriter (\tt)
    - \normalfont
    - \RequirePackage[T1]{fontenc}
---

# Less, but Better?


One of the leading figures in design in the 20th century
was Dieter Rams. His impact on the field cannot be over-stated.
For example, if you look at any 21st century Apple computing
product, you can see the lines and stylings he created in the 1950s:

![](rams.jpg)

Rams' designs were guided by several principles including:

- Good design makes a product understandable
- Good design is as little design as possible
- Back to simplicity. Back to purity. Less, but better.

Here, I apply "less, but better" to software engineering and knowledge engineering.
At the time of this writing, there is something of a gold rush going on where 
everyone is racing to embrace very large and complex models.  I fear that in all
that rush, unless we pay more attention to the basics, we are going to forget
a lot of hard-won lessons about how to structure code and how to explore new problems.
Sure, sometimes new problems will need to more intricate and memory hungry and energy
hungry methods. But always? I think not. 

So let's review the basics, and see how they can be used to build understandable
tools that run very fast. Then you will know enough when to use those tools, or
when to reach for something else that is much more complicated.

These basic are divided in two:

- _software engineering_ (SE): all the things I want intro-to-SE graduate students to know
  about cdding. These tips sub-divide into:
  - team tips
  - system tips
  - scripting tips (here, I use Python for the scripting since many people have some
    understanding of that language).
 - _knowledge engineering_ (KE): all the things I want SE grad students to know about
   explainable AI and analytics.

Everything here will be example based. 
This book will present a small program (under a 1000 lines)
that illustrates many of the tips and tricks I want to show. 

By the way, that code is interesing in its own right.  
`Tiny.py` is 
 explainable multi-objective semi-supervised learner.
If you do not know those terms, just relax. All they mean is that my AI tools
generate tiny models you can read and understand, and that those tools
describe how to reach mulitple goals, and that this is all done with the least
amount of labels on the data. If that sounds complicated, it really isn't
(see below).

\newpage 

# Less is More

The more you code, they more you have to insepct, test, monitor, port, package,
document
and maintain. The less you code, the easier it is to change and update and
fix things. 

Sometimes, the way to code less is to reject functionality
if it means adding much more code for very little relative gain.
For example, many programs have configuration variables that change what the code
does, and those variables can be set from the command-line. 
Some 
libraries let you generating command-line interfaces
direct from the code. One of the nicest is Vladimir Keleshev's
`docopt` tool which builds the interface by parsing  the docstring at top of file.
`docopt` is under 500 lines of code and it works as well as other tools
that are thousads to tens of thousands lines long.

My own alternative to `docopt` is 20 lines long. I present it here as a little
case study is how to code less. Like `docopt`, my `settings` tool parses the
command line from the top-of-file docstring:

