BEGIN {fmt="python"}
NR<=2 { next}
NR==3 { print "\n```"fmt}
/'''/ { COMMENT = 1 - COMMENT 
        print(COMMENT ?  "```\n" : "\n```"fmt); next }
1
END   { if(!COMMENT) print "```" }

