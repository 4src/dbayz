NR<=2 { next}
NR==3 { print "\n```python"}
/'''/ { COMMENT = 1 - COMMENT 
        print(COMMENT ?  "```\n" : "\n```python"); next }
1
END   { if(!COMMENT) print "```" }

