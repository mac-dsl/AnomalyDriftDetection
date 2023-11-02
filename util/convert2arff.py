## @package pyexample
#  Method for converting files to .arff files to use in MOA.
import sys

def out2arff(filepath, classes={0,1}):
    ## Write new .arff file based on .out files
    #  @param filepath: String representing filepath of .out file to be converted
    #  @param classes: set of classes existing in data, for TSB-UAD {0,1}
    filename = filepath.split('.')[0]
    data = []
    with open(filepath, 'r') as input_file:
        for line in input_file:
            data.append(f"{line.strip()},\n")
    header = f"@relation '{filename}'\n\n"
    header += f"@attribute att1 numeric\n"
    header += f"@attribute class {classes}\n\n"
    header += "@data\n\n"
    with open(f"{filename}.arff", 'w') as output_file:
        output_file.writelines([header] + data)

out2arff(sys.argv[1])
