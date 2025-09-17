adhar_number = input("Enter your adhar number: ")

if len(adhar_number) == 12 and adhar_number.isdigit():
    print("Valid Adhar Number")
elif len(adhar_number) < 12 or not adhar_number.isdigit():
    print("Adhar Number is less than 12 digits or contains characters")
elif len(adhar_number) > 12 or not adhar_number.isdigit():
    print("Adhar Number is more than 12 digits or contains characters")