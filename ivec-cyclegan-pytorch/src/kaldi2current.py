import os

CURRENT_PATH = os.getcwd()
target_file="ivector.scp"

if __name__=="__main__":
    with open(target_file) as f:
        data = f.readlines()
    
    os.system("mv %s %s"%(target_file, target_file+".backup"))
    
    with open(target_file,"w") as f:
        for d in data:
            pos1, pos2=d.strip().split()
            name=pos2.split("/")[-1]
            new_line = pos1 +" "+os.path.join(CURRENT_PATH,name)
            print("%s -> %s"%(d.strip(),new_line))
            f.write(new_line+"\n")


