text = "The goal is to turn data into information, and information into insight."

text = text.upper()
text = text.replace(",", "")
text = text.replace(".", "")
text = text.split(" ")
print(text)

a = "d a t a s c i e n c e"
lst = a.upper().split(" ")
print("1: ", len(lst), "\n2: ", lst[0] + lst[10])

newlst = lst[0:4]
print(newlst)

lst.pop(8)
lst.append("Z")
print(lst)

lst.insert(8, "N")
print(lst)

dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

print(dict.keys())

dict.update({"Daisy": ["England", 13]})
print(dict)

dict.update({"Ahmet": ["Turkey", 24]})
print(dict)

dict.pop("Antonio")
print(dict)

l = [2, 13, 18, 93, 22]


def func(a):
    """
    return even and odd list from a list

    Parameters
    ----------
    a = list

    Returns
    -------
    even list from a
    odd list from a
    """
    if type(a) != list:
        return "please enter a list"
    odd = []
    even = []
    for i in a:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return even, odd

evenlist, oddlist = func(l)
print(evenlist, "\n", oddlist)

def add_element(a,b,list):
    c = a*b
    list.append(c)

list = []
add_element(3,5,list)
print(list)


import seaborn as sns

df = sns.load_dataset("car_crashes")
lst = ["NUM_"+ col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]
lst = [col for col in df.columns if df[col].dtype != "O"]

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]

new_df = df[new_cols]
new_df.head()

lst =  [i + "_FLAG" if "NO" not in i else i for i in lst]

print(lst)
