import subprocess
import signal
import time
import os
import sys
start = time.time()
# 生成运行模型的子进程命令
file_path = sys.argv[1]
if os.path.splitext(file_path)[1] == ".py":
    cmd = "python3 " + file_path
else:
    dir_path, file_name = os.path.split(file_path)
    cmd = "cd " + dir_path + ";./" + file_name

print("Start collecting CPU and memory utilization data")
print("Need to run the model, please wait for the model to finish running")
print("Please wait a moment")
print()
# 运行子进程
f = open("ctrlCpuAndMemoryData.txt", "w")
p = subprocess.Popen(cmd,
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     encoding='utf-8',
                     preexec_fn=os.setsid
                     )

time.sleep(0.03)
p1 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                      shell=True,
                      stdout=subprocess.PIPE,
                      stdin=subprocess.PIPE,
                      encoding='utf-8',
                      preexec_fn=os.setsid
                      )
time.sleep(0.1)
p2 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                      shell=True,
                      stdout=subprocess.PIPE,
                      stdin=subprocess.PIPE,
                      encoding='utf-8',
                      preexec_fn=os.setsid
                      )
time.sleep(0.1)
p3 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                      shell=True,
                      stdout=subprocess.PIPE,
                      stdin=subprocess.PIPE,
                      encoding='utf-8',
                      preexec_fn=os.setsid
                      )
time.sleep(0.1)
p4 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                      shell=True,
                      stdout=subprocess.PIPE,
                      stdin=subprocess.PIPE,
                      encoding='utf-8',
                      preexec_fn=os.setsid
                      )
time.sleep(0.1)
p5 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                      shell=True,
                      stdout=subprocess.PIPE,
                      stdin=subprocess.PIPE,
                      encoding='utf-8',
                      preexec_fn=os.setsid
                      )
time.sleep(0.1)
p6 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                      shell=True,
                      stdout=subprocess.PIPE,
                      stdin=subprocess.PIPE,
                      encoding='utf-8',
                      preexec_fn=os.setsid
                      )
f.write(p1.stdout.read() + 'end')

f.write(p2.stdout.read() + 'end')

f.write(p3.stdout.read() + 'end')

f.write(p4.stdout.read() + 'end')

f.write(p5.stdout.read() + 'end')

f.write(p6.stdout.read() + 'end')

p1.kill()
p1.terminate()
p2.kill()
p2.terminate()
p3.kill()
p3.terminate()
p4.kill()
p4.terminate()
p5.kill()
p5.terminate()
p6.kill()
p6.terminate()
while p.poll() is None:

    p1 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                          shell=True,
                          stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          encoding='utf-8',
                          preexec_fn=os.setsid
                          )
    time.sleep(0.1)
    p2 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                          shell=True,
                          stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          encoding='utf-8',
                          preexec_fn=os.setsid
                          )
    time.sleep(0.1)
    p3 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                          shell=True,
                          stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          encoding='utf-8',
                          preexec_fn=os.setsid
                          )
    time.sleep(0.1)
    p4 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                          shell=True,
                          stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          encoding='utf-8',
                          preexec_fn=os.setsid
                          )
    time.sleep(0.1)
    p5 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                          shell=True,
                          stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          encoding='utf-8',
                          preexec_fn=os.setsid
                          )
    time.sleep(0.1)
    p6 = subprocess.Popen(r"npu-smi info -t usages -i 0",
                          shell=True,
                          stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          encoding='utf-8',
                          preexec_fn=os.setsid
                          )
    f.write(p1.stdout.read() + 'end')

    f.write(p2.stdout.read() + 'end')

    f.write(p3.stdout.read() + 'end')

    f.write(p4.stdout.read() + 'end')

    f.write(p5.stdout.read() + 'end')

    f.write(p6.stdout.read() + 'end')

    p1.kill()
    p1.terminate()
    p2.kill()
    p2.terminate()
    p3.kill()
    p3.terminate()
    p4.kill()
    p4.terminate()
    p5.kill()
    p5.terminate()
    p6.kill()
    p6.terminate()

stdout, stderr = p.communicate()
if stderr is not None and stderr != "":
    print(stderr)
    print()
    print(
        "Whether the above error is affecting the CPU and memory utilization (Y/N):",
        end=" ")
    flag = input()
    if flag == "Y":
        print("Please rule out the error and reimplement this step")
        f.write("0")
    else:
        print("Please move on to the next step")
        f.write("1")
else:
    f.write("1")
    print("Collection completed")
f.close()
p.kill()
p.terminate()
p1.kill()
p1.terminate()
