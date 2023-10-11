import pandas as pd

#Reformat Dataset 1
with open('project3_dataset1.txt', 'r') as data:
  plaintext = data.read()
plaintext = plaintext.replace('\t', ',')
plaintext = plaintext.replace(' ', ',')
print(plaintext)
with open('data1.csv', 'w') as f:
  f.write(plaintext)

#Reformat Dataset 2
with open('project3_dataset2.txt', 'r') as data:
  plaintext = data.read()
plaintext = plaintext.replace('\t', ',')
plaintext = plaintext.replace(' ', ',')
print(plaintext)
with open('data2.csv', 'w') as f:
  f.write(plaintext)

