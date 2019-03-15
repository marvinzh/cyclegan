import os

CURRENT_PATH = os.getcwd()
# target_file="ivector.scp"
# target_files = os.listdir()

def file_filter(file):
    if file.split(".")[-1] == "scp":
        return True
    return False

target_files = list(filter(file_filter, os.listdir()))

if __name__=="__main__":

    for target_file in target_files:    
        with open(target_file) as f:
            data = f.readlines()

        os.system("mv %s %s"%(target_file, target_file+".backup"))
        with open(target_file,"w") as f:
            for d in data:
                # pos1, pos2=d.strip().split()
                tokens = d.strip().split()
                addr = tokens[-1]
                name=addr.split("/")[-1]
                new_line = " ".join(tokens[:-1]) +" "+os.path.join(CURRENT_PATH,name)
                print("%s -> %s"%(d.strip(),new_line))
                f.write(new_line+"\n")


