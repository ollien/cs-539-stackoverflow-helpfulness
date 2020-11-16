import pandas as pd
import csv
from itertools import zip_longest

file_name_train = "/Users/miranda/PycharmProjects/Question_id_pull/train.csv"
file_name_test = "/Users/miranda/PycharmProjects/Question_id_pull/valid.csv"

#read files
df_train = pd.read_csv(file_name_train)
df_test = pd.read_csv(file_name_test)

#define lists
train_list = []
test_list = []
train_ID = []
test_ID = []

#append to lists
for i in df_train["Body"]:
    train_list.append(i)

for x in df_test["Body"]:
    test_list.append(x)

for a in df_train["Id"]:
    train_ID.append(a)

for b in df_test["Id"]:
    test_ID.append(b)


#Train File
text_list = []
code_list = []
text_char = []
code_char = []

for i in train_list:
    text_list = []
    code_list = []
    ind = train_list.index(i)
    cleaned_text = i.replace("<p>", "")
    cleaned_text = cleaned_text.replace("</p>", "")

    if "<pre>" and "</pre>" in cleaned_text:
        text_split = cleaned_text.split("<pre>")

        for z in text_split:
            if "</pre>" in z:
                clean = z.replace("</pre","")
                if "<code>" and "</code>" in clean:
                    clean = clean.replace("<code>","")
                    clean = clean.replace("</code>", "")
                    code_list.append(clean)
                else:
                    code_list.append(clean)
            elif "<code>" and "</code>" in z:
                clean = z.replace("<code>", "")
                clean = clean.replace("</code>", "")
                code_list.append(clean)
            else:
                text_list.append(z)
    elif "<code>" and "</code>" in cleaned_text:
        text_split = cleaned_text.split("<code>")
        for y in text_split:
            if "</code>" in text_split:
                clean = y.replace("</code", "")
                if "<pre>" and "</pre>" in clean:
                    clean = clean.replace("<pre>", "")
                    clean = clean.replace("</pre>", "")
                    code_list.append(clean)
                else:
                    code_list.append(clean)
            elif "<pre>" and "</pre>" in y:
                clean = y.replace("<pre>", "")
                clean = clean.replace("</pre>", "")
                code_list.append(clean)
            else:
                text_list.append(y)

    else:
      text_list.append(cleaned_text)



    text = len("".join(text_list))
    code = len("".join(code_list))
    text_char.append(text)
    code_char.append(code)


ratio = 0
ratio_list = []
for element in text_char:
    list_ind = text_char.index(element)
    if code_char[list_ind] == 0:
        ratio = 1
    else:
        ratio = str(element/code_char[list_ind])
    ratio_list.append(ratio)


df = pd.read_csv("/Users/miranda/PycharmProjects/Question_id_pull/train_download.csv")
df["Text"] = text_char
df["Code"] = code_char
df["Text-Code Ratio"] = ratio_list
df.to_csv('Train_Code_Text.csv', index=False)

#Test File

text_list_2 = []
code_list_2 = []
text_char_2 = []
code_char_2 = []
for i2 in test_list:
    text_list_2 = []
    code_list_2 = []
    ind = test_list.index(i2)
    cleaned_text2 = i2.replace("<p>", "")
    cleaned_text2 = cleaned_text2.replace("</p>", "")

    if "<pre>" and "</pre>" in cleaned_text2:
        text_split2 = cleaned_text2.split("<pre>")

        for z2 in text_split2:
            if "</pre>" in z2:
                clean2 = z2.replace("</pre","")
                if "<code>" and "</code>" in clean2:
                    clean2 = clean2.replace("<code>","")
                    clean2 = clean2.replace("</code>", "")
                    code_list_2.append(clean2)
                else:
                    code_list_2.append(clean2)
            elif "<code>" and "</code>" in z2:
                clean2 = z2.replace("<code>", "")
                clean2 = clean2.replace("</code>", "")
                code_list_2.append(clean2)
            else:
                text_list_2.append(z2)
    elif "<code>" and "</code>" in cleaned_text2:
        text_split_2 = cleaned_text2.split("<code>")
        for y2 in text_split_2:
            if "</code>" in text_split_2:
                clean2 = y2.replace("</code", "")
                if "<pre>" and "</pre>" in clean2:
                    clean2 = clean2.replace("<pre>", "")
                    clean2 = clean2.replace("</pre>", "")
                    code_list_2.append(clean2)
                else:
                    code_list_2.append(clean2)
            elif "<pre>" and "</pre>" in y2:
                clean2 = y2.replace("<pre>", "")
                clean2 = clean2.replace("</pre>", "")
                code_list_2.append(clean2)
            else:
                text_list_2.append(y2)

    else:
      text_list_2.append(cleaned_text2)



    text2 = len("".join(text_list_2))
    code2 = len("".join(code_list_2))
    text_char_2.append(text2)
    code_char_2.append(code2)


ratio2 = 0
ratio_list2 = []
for element2 in text_char_2:
    list_ind2 = text_char_2.index(element2)
    if code_char[list_ind2] == 0:
        ratio2 = 1
    else:
        ratio2 = str(element2/code_char[list_ind2])
    ratio_list2.append(ratio2)


df2 = pd.read_csv("/Users/miranda/PycharmProjects/Question_id_pull/test_download.csv")

df2["Text"] = text_char_2
df2["Code"] = code_char_2
df2["Text-Code Ratio"] = ratio_list2

df2.to_csv('Test_Code_Text.csv', index=False)