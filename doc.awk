BEGIN {fmt="python"}
/#include / {rinclude($0)}
NR<=2 { next}
NR==3 { print "\n```"fmt}
/'''/ { COMMENT = 1 - COMMENT 
        print(COMMENT ?  "```\n" : "\n```"fmt); next }
1
END   { if(!COMMENT) print "```" }

function rinclude (line,    x,a) {
   split(line,a,/ /);
   if ( a[1] ~ /^#include/ ) { 
     while ( ( getline x < a[2] ) > 0) rinclude(x);
     close(a[2]) 
     }
   else {print line}
}
