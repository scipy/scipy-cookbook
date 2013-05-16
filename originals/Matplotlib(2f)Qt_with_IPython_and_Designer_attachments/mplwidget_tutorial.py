# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mplwidget_tutorial.ui'
#
# Created: Thu Sep 15 15:59:51 2005
#      by: The PyQt User Interface Compiler (pyuic) 3.13
#
# WARNING! All changes made in this file will be lost!


from qt import *
from mplwidget import *


class Form1(QMainWindow):
    def __init__(self,parent = None,name = None,fl = 0):
        QMainWindow.__init__(self,parent,name,fl)
        self.statusBar()

        if not name:
            self.setName("Form1")


        self.setCentralWidget(QWidget(self,"qt_central_widget"))
        Form1Layout = QVBoxLayout(self.centralWidget(),11,6,"Form1Layout")

        self.matplotlibWidget1 = MatplotlibWidget(self.centralWidget(),"matplotlibWidget1")
        Form1Layout.addWidget(self.matplotlibWidget1)



        self.languageChange()

        self.resize(QSize(422,346).expandedTo(self.minimumSizeHint()))
        self.clearWState(Qt.WState_Polished)


    def languageChange(self):
        self.setCaption(self.__tr("Form1"))


    def __tr(self,s,c = None):
        return qApp.translate("Form1",s,c)
