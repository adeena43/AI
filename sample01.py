def div(dividend, divisor):
    count = 0
    if(divisor==0):
        print("Cannot divide the number")
        break
    for i in range(divisor):
        if(divisor<=dividend):
            dividend = dividend - divisor
            count = count + 1
        else:
            rem = dividend
            break
    print(f"The remiander is: {rem}")
    print(f"The quotient is: {count}")

div(6, 3)
