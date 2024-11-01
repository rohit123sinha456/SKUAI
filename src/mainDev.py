from DataDev import LayoutTextExtractor
from TableDev import TableExtractor
from ColumnDev import ColumnExtractor

text = LayoutTextExtractor()
table = TableExtractor()
column = ColumnExtractor()

def inferFromLayout(path, layout, labels):
   
    table_list = []
    column_list = []
    text_list = []
    label_map = {}
    for key, value in labels.items():
        if value["Type"] == "Table":
            table_list.append(int(key))
        if value["Type"] == "Text":
            text_list.append(int(key))
        if value["Type"] == "Column":
            column_list.append(int(key))
        label_map[int(key)] =  value["Name"]

    extracted_text_data = text.extract_text_from_block(path, layout, label_map, text_list)
    extracted_table_data = table.extract_table_from_block(path, layout, label_map, table_list)
    extracted_column_data = column.extract_column_from_block(path, layout, label_map, column_list)

    myDict = {}

    # Text Data
    for id, data in enumerate(extracted_text_data):
        # print(f"{data['label']}: ", end="")
        wordlist = []
        for word in data['words']:
            wordlist.append(word['text'])
            # print(f"{word['text']} ", end="")
        myDict[data['label']] = ' '.join(wordlist)
        # print()

    # Table Data
    for id, data in enumerate(extracted_table_data):
        myDict[data['label']] = data['table']
    
    # Column Data
    for id, data in enumerate(extracted_column_data):
        myDict[data['label']] = data['columns']

    return myDict

# if __name__ == "__main__":
    
    # labels = {
    #     0 : {
    #         "Name" : "Fax",
    #         "Type" : "Text"
    #     },
    #     1 : {
    #         "Name" : "Name",
    #         "Type" : "Text"
    #     },
    #     2 : {
    #         "Name" : "PODate",
    #         "Type" : "Text"
    #     },
    #     3 : {
    #         "Name" : "PONumber",
    #         "Type" : "Text"
    #     },
    #     4 : {
    #         "Name" : "Supplier",
    #         "Type" : "Text"
    #     },
    #     5 : {
    #         "Name" : "TableContents",
    #         "Type" : "Table"
    #     },
    #     6 : {
    #         "Name" : "Telefone",
    #         "Type" : "Text"
    #     }
    # }

#     myFunction(labels=labels)

# # extracted_text_data = text.extract_text_from_block(path, label_map)
# # extracted_table_data = table.extract_table_from_block(path, label_map)


# # for id, data in enumerate(extracted_text_data):
# #     print(f"{data['label']}: ", end="")
# #     for word in data['words']:
# #         print(f"{word['text']} ", end="")
# #     print()

# # df = pd.read_excel(r"C:/Users/datacore/Downloads/STRAUSS/table.xlsx")
# # print("\n", df)
