from PyDSS.pyContrReader import pyContrReader as PCR
import numpy as np
import pathlib
import logging
import os


class ResultContainer:
    def __init__(self, Options, SystemPaths, dssObjects, dssObjectsByClass, dssBuses, dssSolver):
        LoggerTag = Options['Active Project'] + '_' + Options['Active Scenario']
        self.__dssDolver = dssSolver
        self.Results = {}
        self.CurrentResults = {}
        self.pyLogger = logging.getLogger(LoggerTag)
        self.Buses = dssBuses
        self.ObjectsByElement = dssObjects
        self.ObjectsByClass = dssObjectsByClass
        self.SystemPaths = SystemPaths
        self.__Settings = Options
        self.__StartDay = Options['Start Day']
        self.__EndDay = Options['End Day']
        self.__DateTime = []
        self.FileReader = PCR(SystemPaths['ExportLists'])

        self.ExportFolder = os.path.join(self.SystemPaths['Export'], Options['Active Scenario'])

        pathlib.Path(self.ExportFolder).mkdir(parents=True, exist_ok=True)

        if self.__Settings['Export Mode'] == 'byElement':
            self.ExportList = self.FileReader.pyControllers['ExportMode-byElement']
            self.CreateListByElement()
        elif self.__Settings['Export Mode'] == 'byClass':
            self.ExportList = self.FileReader.pyControllers['ExportMode-byClass']
            self.CreateListByClass()
        return

    def CreateListByClass(self):
        for Class , Properties in self.ExportList.items():
            if Class == 'Buses':
                self.Results[Class] = {}
                self.CurrentResults[Class] = {}
                for PptyIndex, PptyName in Properties.items():
                    if isinstance(PptyName, str):
                        self.Results[Class][PptyName] = {}
                        self.CurrentResults[Class][PptyName] = {}
                        for BusName, BusObj in self.Buses.items():
                            if self.Buses[BusName].inVariableDict(PptyName):
                                self.Results[Class][PptyName][BusName] = []
                                self.CurrentResults[Class][PptyName][BusName] = None
            else:
                if Class in self.ObjectsByClass:
                    self.Results[Class] = {}
                    self.CurrentResults[Class] = {}
                    for PptyIndex, PptyName in Properties.items():
                        if isinstance(PptyName, str):
                            self.Results[Class][PptyName] = {}
                            self.CurrentResults[Class][PptyName] = {}
                            for ElementName, ElmObj in self.ObjectsByClass[Class].items():
                                if self.ObjectsByClass[Class][ElementName].IsValidAttribute(PptyName):
                                    self.Results[Class][PptyName][ElementName] = []
                                    self.CurrentResults[Class][PptyName][ElementName] = None
        return

    def CreateListByElement(self):
        for Element, Properties in self.ExportList.items():
            if Element in self.ObjectsByElement:
                self.Results[Element] = {}
                self.CurrentResults[Element] = {}
                for PptyIndex, PptyName in Properties.items():
                    if isinstance(PptyName, str):
                        if self.ObjectsByElement[Element].IsValidAttribute(PptyName):
                            self.Results[Element][PptyName] = []
                            self.CurrentResults[Element][PptyName] = None
            elif Element in self.Buses:
                self.Results[Element] = {}
                self.CurrentResults[Element] = {}
                for PptyIndex, PptyName in Properties.items():
                    if isinstance(PptyName, str):
                        if self.Buses[Element].inVariableDict(PptyName):
                            self.Results[Element][PptyName] = []
                            self.CurrentResults[Element][PptyName] = None
        return

    def UpdateResults(self):
        self.__DateTime.append(self.__dssDolver.GetDateTime())
        if self.__Settings['Export Mode'] == 'byElement':
            for Element in self.Results.keys():
                for Property in self.Results[Element].keys():
                    if '.' in Element:
                        self.Results[Element][Property].append(self.ObjectsByElement[Element].GetValue(Property))
                        self.CurrentResults[Element][Property] = self.ObjectsByElement[Element].GetValue(Property)
                    else:
                        self.Results[Element][Property].append(self.Buses[Element].GetVariable(Property))
                        self.CurrentResults[Element][Property] = self.Buses[Element].GetVariable(Property)
        elif self.__Settings['Export Mode'] == 'byClass':
            for Class in self.Results.keys():
                for Property in self.Results[Class].keys():
                    for Element in self.Results[Class][Property].keys():
                        if Class == 'Buses':
                            self.Results[Class][Property][Element].append(self.Buses[Element].GetVariable(Property))
                            self.CurrentResults[Class][Property][Element] = self.Buses[Element].GetVariable(Property)
                        else:
                            self.Results[Class][Property][Element].append(
                                self.ObjectsByClass[Class][Element].GetValue(Property))
                            self.CurrentResults[Class][Property][Element] = \
                                self.ObjectsByClass[Class][Element].GetValue(Property)
        return

    def ExportResults(self):
        if self.__Settings['Export Mode'] == 'byElement':
            self.__ExportResultsByElements()
        elif self.__Settings['Export Mode'] == 'byClass':
            self.__ExportResultsByClass()

    def __ExportResultsByClass(self):
        for Class in self.Results.keys():
            for Property in self.Results[Class].keys():
                Class_ElementDatasets = []
                PptyLvlHeader = ''
                for Element in self.Results[Class][Property].keys():
                    ElmLvlHeader = ''
                    if isinstance(self.Results[Class][Property][Element][0], list):
                        Data = np.array(self.Results[Class][Property][Element])
                        for i in range(len(self.Results[Class][Property][Element][0])):
                            ElmLvlHeader += Element + '-' + str(i) + ','
                    else:
                        Data = np.transpose(np.array([self.Results[Class][Property][Element]]))
                        ElmLvlHeader = Element + ','

                    if self.__Settings['Export Style'] == 'Separate files':
                        fname = '-'.join([Class, Property, Element, str(self.__StartDay), str(self.__EndDay)]) + '.csv'
                        np.savetxt(os.path.join(self.ExportFolder, fname), Data,
                                   delimiter=',', header=ElmLvlHeader, comments='', fmt='%f')
                        self.pyLogger.info(Class + '-' + Property  + '-' + Element + ".csv exported to " + self.ExportFolder)
                    elif self.__Settings['Export Style'] == 'Single file':
                        Class_ElementDatasets.append(Data)
                    PptyLvlHeader += ElmLvlHeader
                if self.__Settings['Export Style'] == 'Single file':
                    Dataset = Class_ElementDatasets[0]
                    if len(Class_ElementDatasets) > 0:
                        for D in Class_ElementDatasets[1:]:
                            Dataset = np.append(Dataset, D, axis=1)
                    fname = '-'.join([Class, Property, str(self.__StartDay), str(self.__EndDay)]) + '.csv'
                    np.savetxt(os.path.join(self.ExportFolder, fname), Dataset,
                               delimiter=',', header=PptyLvlHeader, comments='', fmt='%f')
                    self.pyLogger.info(Class + '-' + Property + ".csv exported to " + self.ExportFolder)
        return

    def __ExportResultsByElements(self):
        for Element in self.Results.keys():
            ElementDatasets = []
            AllHeader = ''

            for Property in self.Results[Element].keys():
                Header = ''

                if isinstance(self.Results[Element][Property][0], list):
                    Data = np.array(self.Results[Element][Property])
                    for i in range(len(self.Results[Element][Property][0])):
                        Header += Property + '-' + str(i) + ','
                else:
                    Data = np.transpose(np.array([self.Results[Element][Property]]))
                    Header = Property + ','

                if self.__Settings['Export Style'] == 'Separate files':
                    fname = '-'.join([Element, Property, str(self.__StartDay), str(self.__EndDay)]) + '.csv'
                    np.savetxt(os.path.join(self.ExportFolder, fname), Data,
                               delimiter=',', header=Header, comments='', fmt='%f')
                    self.pyLogger.info(Element + '-' + Property + ".csv exported to " + self.ExportFolder)
                elif self.__Settings['Export Style'] == 'Single file':
                    ElementDatasets.append(Data)
                AllHeader += Header
            if self.__Settings['Export Style'] == 'Single file':
                Dataset = ElementDatasets[0]
                if len(ElementDatasets) > 0:
                    for D in ElementDatasets[1:]:
                        Dataset = np.append(Dataset, D, axis=1)
                fname = '-'.join([Element, str(self.__StartDay), str(self.__EndDay)]) + '.csv'
                np.savetxt(os.path.join(self.ExportFolder, fname), Dataset,
                           delimiter=',', header=AllHeader, comments='', fmt='%f')
                self.pyLogger.info(Element + ".csv exported to " + self.ExportFolder)
        return
