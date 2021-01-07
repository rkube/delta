#!/usr/bin python

#Reads in log files from delta/tests/test_executor.py.
#LogParser file is general, can be imported in other scripts
# (e.g. "from read_executor import LogParser")
#if this script run as a main file, supply the log file as an argument:
#   python read_executor.py slurm-3771414.out
```
import re
import time,datetime

class LogParser():
    def __init__(self,filename):
        self.filename = filename
        f = open(filename)
        self.lines = f.readlines()
        #sort lines by tstep
        #self.lines.sort(key=lambda r:re.findall('tstep=(\d+)',r))

    def parse(self):
        dts = []; tis = []; ranks = []
        t0 = None
        for line in self.lines:
            if 'INFO' in line and t0 is None:
                t0 = datetime.datetime.strptime(line.split()[0],"%H:%M:%S,%f")
            if 'perform_analysis done' in line:
                dt = float(re.findall('time elapsed: (\d+)',line)[0])
                ti = (datetime.datetime.strptime(line.split()[0],"%H:%M:%S,%f") - t0).total_seconds() - dt
                rank = int(re.findall('rank=(\d+)',line)[0])

                dts.append(dt)
                tis.append(ti)
                ranks.append(rank)
        return tis,dts,ranks

if __name__=="__main__":
    import matplotlib.pyplot as plt; plt.ion()
    import sys

    filename = sys.argv[1]
    logp = LogParser(filename)
    tis,dts,ranks = logp.parse()

    for (ti,dt,rank) in zip(tis,dts,ranks):
        plt.plot([ti,ti+dt],[rank,rank],'g')
