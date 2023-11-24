import getopt
import sys

def main(argv):
    # Define the command-line options and default values
    input_file = ''
    output_file = ''
    verbose = False

    # Define the usage message
    usage = 'Usage: script.py -i <inputfile> -o <outputfile> [-v]'

    try:
        # Parse the command-line arguments
        opts, args = getopt.getopt(argv, "hi:o:v", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt == '-v':
            verbose = True

    # Process the input and output files
    print('Input file is "', input_file, '"')
    print('Output file is "', output_file, '"')
    if verbose:
        print("Verbose mode is enabled")

if __name__ == "__main__":
   main(sys.argv[1:])
