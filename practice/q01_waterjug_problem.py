a = int(input("Enter Jug A Capacity: "))
b = int(input("Enter Jug B Capacity: "))
ai = int(input("Initally water in Jug A: "))
bi = int(input("Initally water in Jug B: "))
af = int(input("Final state of Jug A: "))
bf = int(input("Final state of Jug B: "))

print("\nList of operations you can do: \n")
print("1. Fill Jug A Completely\n")
print("2. Fill Jug B Completely\n")
print("3. Empty Jug A Completely\n")
print("4. Empty Jug B Completely\n")
print("5. Pour from Jug A till Jug B is filled completely or Jug A becomes empty.\n")
print("6. Pour from Jug B till Jug A is filled completely or Jug B becomes empty.\n")
print("7. Pour all from Jug B to Jug A.\n")
print("8. Pour all from Jug A to Jug B.\n")

while ((ai != af or bi != bf)):
    op = int(input("Enter the operation: "))
    if (op==1):
        ai = a
    elif (op==2):
        bi = b
    elif (op==3):
        ai=0
    elif (op==4):
        bi=0
    elif (op==5):
        if (b-bi > ai):
            bi = ai + bi
            ai = 0
        else:
            ai = ai-(b-bi)
            bi = b
    elif (op==6):
        if (a-ai>bi):
            ai = ai+bi
            bi=0
        else:
            bi = bi-(a-ai)
            ai = a
    elif (op==7):
        ai = ai+bi
        bi = 0
    elif (op==8):
        bi = bi+ai
        ai = 0

print("Final state of Jug A: ",ai," and, Jug B: ",bi," has been reached.")
