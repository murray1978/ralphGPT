import os

# Data files
data_folder = "datasets/"
datafile = "datasets/dataset/ijcnlp_dailydialog/dialogues_text.txt"
outputfile = data_folder + "dialogues.txt"

# special tokens.
eou_token = "__eou__"
usr_token = "__usr__"
bot_token = "__bot__"

# lets load the parameters, TODO - maybe here for a while.....
if os.path.exists(datafile):
    print(f"json file {datafile} found, loading")
else:
    print(f"json file {datafile} not found")
    exit()

# Open up the inputfile for the tokenizer
with open(datafile, 'r', encoding='utf-8') as f:
    text = f.read()




# exmple string
# __usr__ hello Ralph. __eou__ __bot__ hello how are you? __eou__ __usr__ I am fine. __eou__
# Open output file for writing
with open(outputfile, 'w', encoding='utf-8') as out_file:
    output = ""
    isUser = True

    # Process each line separately to maintain new lines and add __eoc__ token
    for line in text.splitlines():

        # Start each line with __usr__ or __bot__ based on the initial speaker
        output += usr_token + " "
        isUser = True  # Start with the user as the initial speaker

        words = line.split()  # Split each line into words
        for i, word in enumerate(words):
            if word == eou_token:
                isUser = not isUser  # Toggle the speaker
                if i < len(words) - 1:
                    output += " " + eou_token + " " + (usr_token if isUser else bot_token) + " "
            else:
                output += " " + word

        # Add the __eoc__ token at the end of each conversation line and a newline
        output += " __eoc__\n"

    # Write the final output to the file
    out_file.write(output)


