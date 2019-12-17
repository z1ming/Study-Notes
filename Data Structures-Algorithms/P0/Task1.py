"""
Read file into texts and calls.
It's ok if you don't understand how to read files.
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

res = []
for i in texts:
    res.append(i[0].replace(" ",""))
    res.append(i[1].replace(" ",""))
n = len(set(res))

res1 = []
for j in calls:
    res1.append(j[0].replace(" ",""))
    res1.append(j[1].replace(" ",""))
m = len(set(res1))

print("There are {} different telephone numbers in the records.".format(n+m))

"""
TASK 1:
How many different telephone numbers are there in the records? 
Print a message:
"There are <count> different telephone numbers in the records."
"""
# 时间复杂度：O(n)
# 空间复杂度：O(n)