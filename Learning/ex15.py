from sys import argv

script, filename = argv

txt = open(filename)

print "Hhere's your file %r:" %filename
print txt.read()

print "Type the fiilename again:"

file_again = raw_input(">")

txt_again = open(file_again)

print txt_again.read()


