import sys
class multi_pbars:
    def __init__(self,bar_string,l = 40):
        """Maintain multiple progress bars of chiron running
        Args:
            bar_string([string]): List of the names of the bars.
            progress([int/float]): 
        """
        if isinstance(bar_string,str):
            bar_string = [bar_string]
        elif not isinstance(bar_string,list):
            raise ValueError("Bar string must be either string type or list type")
        self.bar_string = bar_string
        self.bar_n = len(bar_string)
        self.progress = [0]*self.bar_n
        self.total = [-1]*self.bar_n
        self.max_line = 0
        self.bar_l = l
    def update(self,i,progress=None,total=None,title = None):
        if progress is not None:
            self.progress[i] = progress
        if total is not None:
            self.total[i] = total
        if title is not None:
            self.bar_string[i] = title
    def update_bar(self):
        self.refresh()
    def refresh(self):
        text = '\r'
        for i in range(self.bar_n):
            p = float(self.progress[i])/(self.total[i]+1e-6)
            if p>1:
                p=1
            elif p<0:
                p=0
            block = int(round(p*self.bar_l))
            current_line = "%s: %5.1f%%|%s| %d/%d"%( self.bar_string[i],p*100,"#"*block + "-"*(self.bar_l-block), self.progress[i],self.total[i])
            self.max_line = max(len(current_line),self.max_line)
            text += current_line + ' '*(self.max_line-len(current_line)) + '\n'
        text += '\033[%dA'%(self.bar_n)
        sys.stdout.write(text)
        sys.stdout.flush()
    def end(self):
        text = '\n'*self.bar_n
        sys.stdout.write(text)
        sys.stdout.flush()
if __name__== "__main__":
    import time
    test_bars = multi_pbars(['title1','title2','title3'])
    s_time = time.time()
    for i in range(100):
        time.sleep(0.1)
        for j in range(2):
            test_bars.update(j,100-i,100)
        test_bars.refresh()
    e_time = time.time()
    sys.stdout.write("\n\n\nTime consumed:%f\n"%(e_time-s_time))
    sys.stdout.flush()
