# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:50:21 2022

@author: Victor Levy dit Vehel, victor.levy.vehel [at] gmail [dot] com
"""

import datetime

class log:
    
    def __init__(self, path):
        self.path = path
        self.print = True
        self.shush = True
        self.info('Log start')
    

    def info(self, message):
        """ shortcut for info messages """
        self.write(message, 'INF0')
        
        
    def warning(self, message):
        """ shortcut for warning messages """
        self.write(message, 'WAR ')
        
        
    def error(self, message):
        """ shortcut for error messages """
        self.write(message, 'ERR ')
      
                
    def write(self, message, level):
        """ format and write the line to the log file. if self.print, send it to term as well """
        line = "[{}][{}] : {}".format(self.timestamp(), level, message)
        if self.print:
            print(line, flush=True)
        if not self.shush:
            with open(self.path, 'a') as logfile:
                logfile.write(line)
                logfile.write('\n')
    
    
    def timestamp(self):
        """ return a YMD - HMS timestamp """
        return " - ".join(datetime.datetime.now().isoformat().split('.')[0].split('T'))
    